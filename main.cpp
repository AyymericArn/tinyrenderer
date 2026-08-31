#include <cmath>
#include "tgaimage.h"
#include <iostream>

constexpr TGAColor white   = {255, 255, 255, 255}; // attention, BGRA order
constexpr TGAColor green   = {  0, 255,   0, 255};
constexpr TGAColor red     = {  0,   0, 255, 255};
constexpr TGAColor blue    = {255, 128,  64, 255};
constexpr TGAColor yellow  = {  0, 200, 255, 255};

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

void triangleBBoxZ(int ax, int ay, int az, int bx, int by, int bz, int cx, int cy, int cz, TGAImage &framebuffer) {
    int bbminx = std::min(std::min(ax, bx), cx); // bounding box for the triangle
    int bbminy = std::min(std::min(ay, by), cy); // defined by its top left and bottom right corners
    int bbmaxx = std::max(std::max(ax, bx), cx);
    int bbmaxy = std::max(std::max(ay, by), cy);

    double total_area = signed_triangle_area(ax, ay, bx, by, cx, cy);

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
            TGAColor color = {
                static_cast<std::uint8_t>(static_cast<int>(alpha * 255.)),
                static_cast<std::uint8_t>(static_cast<int>(beta * 255.)),
                static_cast<std::uint8_t>(static_cast<int>(gamma * 255.)),
                1
            };
            std::clog << "color: " << static_cast<int>(alpha * 255.) << std::endl;
            // std::clog << "alpha: " << alpha << std::endl;
            // std::clog << "beta: " << beta << std::endl;
            // std::clog << "gamma: "<< gamma << std::endl;
            // std::clog << "total: "<< total_area << std::endl;
            framebuffer.set(x, y, color);
        }
    }
}

int main(int argc, char** argv) {
    constexpr int width  = 64;
    constexpr int height = 64;
    TGAImage framebuffer(width, height, TGAImage::GRAYSCALE);

    int ax = 17, ay =  4, az =  13;
    int bx = 55, by = 39, bz = 128;
    int cx = 23, cy = 59, cz = 255;

    triangleBBoxZ(ax, ay, az, bx, by, bz, cx, cy, cz, framebuffer);

    // triangleBBox(  7, 45, 35, 100, 45,  60, framebuffer, red);
    // triangleBBox(120, 35, 90,   5, 45, 110, framebuffer, white);
    // triangleBBox(115, 83, 80,  90, 85, 120, framebuffer, green);

    framebuffer.write_tga_file("framebuffer.tga");
    return 0;
}

