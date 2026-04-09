#include "image_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int almost_equal(float lhs, float rhs, float epsilon) {
    return fabsf(lhs - rhs) <= epsilon;
}

static int test_convolution_identity(void) {
    static const float kernel[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
    };
    const uint8_t src[9] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    float dst[9] = {0};

    if (csrc_convolve3x3_gray_u8(src, 3, 3, 3, kernel, dst, 3) != CSRC_STATUS_OK) {
        return 0;
    }

    return almost_equal(dst[0], 1.0f, 1e-4f) &&
           almost_equal(dst[4], 5.0f, 1e-4f) &&
           almost_equal(dst[8], 9.0f, 1e-4f);
}

static int test_sobel_detects_vertical_edge(void) {
    const uint8_t src[25] = {
        0, 0, 255, 255, 255,
        0, 0, 255, 255, 255,
        0, 0, 255, 255, 255,
        0, 0, 255, 255, 255,
        0, 0, 255, 255, 255,
    };
    float grad_x[25] = {0};
    float grad_y[25] = {0};
    float magnitude[25] = {0};

    if (csrc_sobel_gray_u8(src, 5, 5, 5, grad_x, 5, grad_y, 5, magnitude, 5) != CSRC_STATUS_OK) {
        return 0;
    }

    return magnitude[(2 * 5) + 2] > 500.0f &&
           fabsf(grad_y[(2 * 5) + 2]) < 1e-4f &&
           magnitude[(2 * 5) + 0] < 1e-4f;
}

static int test_laplace_impulse(void) {
    const uint8_t src[9] = {
        0, 0, 0,
        0, 255, 0,
        0, 0, 0,
    };
    float dst[9] = {0};

    if (csrc_laplace_gray_u8(src, 3, 3, 3, dst, 3) != CSRC_STATUS_OK) {
        return 0;
    }

    return almost_equal(dst[4], -1020.0f, 1e-4f) &&
           almost_equal(dst[1], 255.0f, 1e-4f) &&
           almost_equal(dst[3], 255.0f, 1e-4f) &&
           almost_equal(dst[5], 255.0f, 1e-4f) &&
           almost_equal(dst[7], 255.0f, 1e-4f);
}

static int test_harris_corner_response(void) {
    uint8_t src[64];
    float response[64] = {0};
    float max_corner = 0.0f;
    float flat_region = 0.0f;
    float edge_region = 0.0f;

    memset(src, 0, sizeof(src));

    for (size_t y = 3; y < 8; ++y) {
        for (size_t x = 3; x < 8; ++x) {
            src[(y * 8) + x] = 255;
        }
    }

    if (csrc_harris_response_gray_u8(src, 8, 8, 8, 0.04f, response, 8) != CSRC_STATUS_OK) {
        return 0;
    }

    for (size_t y = 2; y <= 4; ++y) {
        for (size_t x = 2; x <= 4; ++x) {
            float value = response[(y * 8) + x];
            if (value > max_corner) {
                max_corner = value;
            }
        }
    }

    flat_region = response[(0 * 8) + 0];
    edge_region = response[(2 * 8) + 4];

    return max_corner > 0.0f &&
           max_corner > flat_region &&
           max_corner > edge_region;
}

static int test_alpha_blend(void) {
    uint8_t background[4 * 4 * 3];
    uint8_t overlay[2 * 2 * 4];
    memset(background, 0, sizeof(background));
    memset(overlay, 0, sizeof(overlay));

    for (size_t i = 0; i < 4 * 4; ++i) {
        background[(i * 3) + 0] = 10;
        background[(i * 3) + 1] = 20;
        background[(i * 3) + 2] = 30;
    }

    for (size_t i = 0; i < 4; ++i) {
        overlay[(i * 4) + 2] = 200;
        overlay[(i * 4) + 3] = 128;
    }

    if (csrc_alpha_blend_bgra_over_bgr(overlay, 2, 2, 8, background, 4, 4, 12, 1, 1) != CSRC_STATUS_OK) {
        return 0;
    }

    return background[((1 * 4 + 1) * 3) + 0] == 5 &&
           background[((1 * 4 + 1) * 3) + 1] == 10 &&
           background[((1 * 4 + 1) * 3) + 2] == 115 &&
           background[((0 * 4 + 0) * 3) + 2] == 30;
}

int main(void) {
    struct {
        const char *name;
        int (*fn)(void);
    } tests[] = {
        {"convolution_identity", test_convolution_identity},
        {"sobel_detects_vertical_edge", test_sobel_detects_vertical_edge},
        {"laplace_impulse", test_laplace_impulse},
        {"harris_corner_response", test_harris_corner_response},
        {"alpha_blend", test_alpha_blend},
    };

    size_t failures = 0;

    for (size_t i = 0; i < sizeof(tests) / sizeof(tests[0]); ++i) {
        int passed = tests[i].fn();
        printf("[%s] %s\n", passed ? "PASS" : "FAIL", tests[i].name);
        if (!passed) {
            ++failures;
        }
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
