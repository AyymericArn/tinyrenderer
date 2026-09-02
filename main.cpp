#include <limits>
#include <algorithm>
#include <cmath>
#include <tuple>
#include "geometry.h"
#include "model.h"
#include "tgaimage.h"
#include <iostream>

constexpr int width  = 800;
constexpr int height = 800;

mat<4,4> ModelView, Viewport, Perspective;

// float fzbuffer[width][height] = {0};

constexpr TGAColor white   = {255, 255, 255, 255}; // attention, BGRA order
constexpr TGAColor green   = {  0, 255,   0, 255};
constexpr TGAColor red     = {  0,   0, 255, 255};
constexpr TGAColor blue    = {255, 128,  64, 255};
constexpr TGAColor yellow  = {  0, 200, 255, 255};

// drawing functions

void line(int ax, int ay, int bx, int by, TGAImage &framebuffer, TGAColor color) {
    bool steep = std::abs(ax-bx) < std::abs(ay-by);
    if (steep) { // if the line is steep, we transpose the image
        std::swap(ax, ay);
        std::swap(bx, by);
    }
    if (ax>bx) { // make it left−to−right
        std::swap(ax, bx);
        std::swap(ay, by);
    }
    float y = ay;
    for (int x=ax; x<=bx; x++) {
        if (steep) // if transposed, de−transpose
            framebuffer.set(y, x, color);
        else
            framebuffer.set(x, y, color);
        y += (by-ay) / static_cast<float>(bx-ax);
    }
}


void triangleScanline(int ax, int ay, int bx, int by, int cx, int cy, TGAImage &framebuffer, TGAColor color) {
    // sort the vertices, a,b,c in ascending y order (bubblesort yay!)
    if (ay>by) { std::swap(ax, bx); std::swap(ay, by); }
    if (ay>cy) { std::swap(ax, cx); std::swap(ay, cy); }
    if (by>cy) { std::swap(bx, cx); std::swap(by, cy); }
    int total_height = cy-ay;

    if (ay != by) { // if the bottom half is not degenerate
        int segment_height = by - ay;
        for (int y=ay; y<=by; y++) { // sweep the horizontal line from ay to by
            int x1 = ax + ((cx - ax)*(y - ay)) / total_height;
            int x2 = ax + ((bx - ax)*(y - ay)) / segment_height;
            for (int x=std::min(x1,x2); x<std::max(x1,x2); x++)  // draw a horizontal line
                framebuffer.set(x, y, color);
        }
    }
    if (by != cy) { // if the upper half is not degenerate
        int segment_height = cy - by;
        for (int y=by; y<=cy; y++) { // sweep the horizontal line from by to cy
            int x1 = ax + ((cx - ax)*(y - ay)) / total_height;
            int x2 = bx + ((cx - bx)*(y - by)) / segment_height;
            for (int x=std::min(x1,x2); x<std::max(x1,x2); x++)  // draw a horizontal line
                framebuffer.set(x, y, color);
        }
    }
}

double signed_triangle_area(int ax, int ay, int bx, int by, int cx, int cy) {
    return .5*((by-ay)*(bx+ax) + (cy-by)*(cx+bx) + (ay-cy)*(ax+cx));
}

void triangleBBoxZ(int ax, int ay, int az, int bx, int by, int bz, int cx, int cy, int cz, TGAImage &zbuffer, TGAImage &framebuffer, TGAColor color) {
    int bbminx = std::min(std::min(ax, bx), cx); // bounding box for the triangle
    int bbminy = std::min(std::min(ay, by), cy); // defined by its top left and bottom right corners
    int bbmaxx = std::max(std::max(ax, bx), cx);
    int bbmaxy = std::max(std::max(ay, by), cy);

    double total_area = signed_triangle_area(ax, ay, bx, by, cx, cy);
    if (total_area<1) return; // backface culling + discarding triangles that cover less than a pixel

#pragma omp parallel for
    for (int x=bbminx; x<=bbmaxx; x++) {
        for (int y=bbminy; y<=bbmaxy; y++) {
            double alpha = signed_triangle_area(x, y, bx, by, cx, cy) / total_area;
            double beta  = signed_triangle_area(x, y, cx, cy, ax, ay) / total_area;
            double gamma = signed_triangle_area(x, y, ax, ay, bx, by) / total_area;
            if (alpha<0 || beta<0 || gamma<0) {
                continue; // negative barycentric coordinate => the pixel is outside the triangle
            }
            unsigned char z = static_cast<unsigned char>(alpha * az + beta * bz + gamma * cz);
            // std::clog << "alpha: " << alpha << std::endl;
            // std::clog << "beta: " << beta << std::endl;
            // std::clog << "gamma: "<< gamma << std::endl;
            // std::clog << "total: "<< total_area << std::endl;
            if (z <= zbuffer.get(x, y)[0]) continue;
            zbuffer.set(x, y, {z});
            framebuffer.set(x, y, color);
        }
    }
}

void rasterize(const vec4 clip[3], std::vector<double> &zbuffer, TGAImage &framebuffer, const TGAColor color) {
    vec4 ndc[3]    = { clip[0]/clip[0].w, clip[1]/clip[1].w, clip[2]/clip[2].w };                // normalized device coordinates
    vec2 screen[3] = { (Viewport*ndc[0]).xy(), (Viewport*ndc[1]).xy(), (Viewport*ndc[2]).xy() }; // screen coordinates

    mat<3,3> ABC = {{ {screen[0].x, screen[0].y, 1.}, {screen[1].x, screen[1].y, 1.}, {screen[2].x, screen[2].y, 1.} }};
    if (ABC.det()<1) return; // backface culling + discarding triangles that cover less than a pixel

    auto [bbminx,bbmaxx] = std::minmax({screen[0].x, screen[1].x, screen[2].x}); // bounding box for the triangle
    auto [bbminy,bbmaxy] = std::minmax({screen[0].y, screen[1].y, screen[2].y}); // defined by its top left and bottom right corners
#pragma omp parallel for
    for (int x=std::max<int>(bbminx, 0); x<=std::min<int>(bbmaxx, framebuffer.width()-1); x++) { // clip the bounding box by the screen
        for (int y=std::max<int>(bbminy, 0); y<=std::min<int>(bbmaxy, framebuffer.height()-1); y++) {
            vec3 bc = ABC.invert_transpose() * vec3{static_cast<double>(x), static_cast<double>(y), 1.}; // barycentric coordinates of {x,y} w.r.t the triangle
            if (bc.x<0 || bc.y<0 || bc.z<0) continue;                                                    // negative barycentric coordinate => the pixel is outside the triangle
            double z = bc * vec3{ ndc[0].z, ndc[1].z, ndc[2].z };
            if (z <= zbuffer[x+y*framebuffer.width()]) continue;
            zbuffer[x+y*framebuffer.width()] = z;
            framebuffer.set(x, y, color);
        }
    }
}

// geometry computation

std::tuple<int,int,int> project(vec3 v) { // First of all, (x,y) is an orthogonal projection of the vector (x,y,z).
    return { (v.x + 1.) *  width/2,       // Second, since the input models are scaled to have fit in the [-1,1]^3 world coordinates,
             (v.y + 1.) * height/2,       // we want to shift the vector (x,y) and then scale it to span the entire screen.
             (v.z + 1.) *   255./2 };
}

vec3 rot(vec3 v) {
    constexpr double a = M_PI/6;
    constexpr mat<3,3> Ry = {{{std::cos(a), 0, std::sin(a)}, {0,1,0}, {-std::sin(a), 0, std::cos(a)}}};
    return Ry*v;
}

vec3 persp(vec3 v) {
    constexpr double c = 3.;
    return v / (1-v.z/c);
}

// with matrices
void viewport(const int x, const int y, const int w, const int h) {
    Viewport = {{{w/2., 0, 0, x+w/2.}, {0, h/2., 0, y+h/2.}, {0,0,1,0}, {0,0,0,1}}};
}

void perspective(const double f) {
    Perspective = {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {0,0, -1/f,1}}};
}

void lookat(const vec3 eye, const vec3 center, const vec3 up) {
    vec3 n = normalized(eye-center);
    vec3 l = normalized(cross(up,n));
    vec3 m = normalized(cross(n, l));
    ModelView = mat<4,4>{{{l.x,l.y,l.z,0}, {m.x,m.y,m.z,0}, {n.x,n.y,n.z,0}, {0,0,0,1}}} *
                mat<4,4>{{{1,0,0,-center.x}, {0,1,0,-center.y}, {0,0,1,-center.z}, {0,0,0,1}}};
}

int main(int argc, char** argv) {
    Model model("./obj/diablo3_pose/diablo3_pose.obj");
    TGAImage framebuffer(width, height, TGAImage::RGBA);
    // TGAImage     zbuffer(width, height, TGAImage::GRAYSCALE);
    std::vector<double> zbuffer(width*height, -std::numeric_limits<double>::max());


    int ax = 17, ay =  4, az =  13;
    int bx = 55, by = 39, bz = 128;
    int cx = 23, cy = 59, cz = 255;

    // triangleBBoxZ(ax, ay, az, bx, by, bz, cx, cy, cz, zbuffer, framebuffer);

    // triangleBBox(  7, 45, 35, 100, 45,  60, framebuffer, red);
    // triangleBBox(120, 35, 90,   5, 45, 110, framebuffer, white);
    // triangleBBox(115, 83, 80,  90, 85, 120, framebuffer, green);

    constexpr vec3    eye{-1,0,2}; // camera position
    constexpr vec3 center{0,0,0};  // camera direction
    constexpr vec3     up{0,1,0};  // camera up vector

    lookat(eye, center, up);                              // build the ModelView   matrix
    perspective(norm(eye-center));                        // build the Perspective matrix
    viewport(width/16, height/16, width*7/8, height*7/8); // build the Viewport    matrix

    // draw model
    // for (int i=0; i<model.nfaces(); i++) { // iterate through all triangles
    //     auto [ax, ay, az] = project(persp(rot(model.vert(i, 0))));
    //     auto [bx, by, bz] = project(persp(rot(model.vert(i, 1))));
    //     auto [cx, cy, cz] = project(persp(rot(model.vert(i, 2))));
    //     TGAColor rnd;
    //     for (int c=0; c<3; c++) rnd[c] = std::rand()%255;
    //     triangleBBoxZ(ax, ay, az, bx, by, bz, cx, cy, cz, zbuffer, framebuffer, rnd);
    // }

    // for (int m=1; m<argc; m++) { // iterate through all input objects
        // Model model(argv[m]);
        for (int i=0; i<model.nfaces(); i++) { // iterate through all triangles
            vec4 clip[3];
            for (int d : {0,1,2}) {            // assemble the primitive
                vec3 v = model.vert(i, d);
                clip[d] = Perspective * ModelView * vec4{v.x, v.y, v.z, 1.};
            }
            TGAColor rnd;
            for (int c=0; c<3; c++) rnd[c] = std::rand()%255;
            rasterize(clip, zbuffer, framebuffer, rnd); // rasterize the primitive
        }
    // }

    framebuffer.write_tga_file("framebuffer.tga");
    // zbuffer.write_tga_file("zbuffer.tga");
    return 0;
}

