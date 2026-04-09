#include "image_ops.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static size_t clamp_index(ptrdiff_t value, size_t limit) {
    if (value < 0) {
        return 0;
    }
    if ((size_t)value > limit) {
        return limit;
    }
    return (size_t)value;
}

static int validate_gray_args(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    const float *dst,
    size_t dst_stride
) {
    if (src == NULL || dst == NULL) {
        return CSRC_STATUS_NULL_PTR;
    }
    if (width == 0 || height == 0 || src_stride < width || dst_stride < width) {
        return CSRC_STATUS_BAD_DIMENSIONS;
    }
    return CSRC_STATUS_OK;
}

static float sample_gray_u8_clamped(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    ptrdiff_t x,
    ptrdiff_t y
) {
    size_t sx = clamp_index(x, width - 1);
    size_t sy = clamp_index(y, height - 1);
    return (float)src[(sy * src_stride) + sx];
}

static float sample_gray_f32_clamped(
    const float *src,
    size_t width,
    size_t height,
    size_t src_stride,
    ptrdiff_t x,
    ptrdiff_t y
) {
    size_t sx = clamp_index(x, width - 1);
    size_t sy = clamp_index(y, height - 1);
    return src[(sy * src_stride) + sx];
}

static float apply_kernel3x3(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    size_t x,
    size_t y,
    const float kernel[9]
) {
    float sum = 0.0f;

    for (ptrdiff_t ky = -1; ky <= 1; ++ky) {
        for (ptrdiff_t kx = -1; kx <= 1; ++kx) {
            float sample = sample_gray_u8_clamped(src, width, height, src_stride, (ptrdiff_t)x + kx, (ptrdiff_t)y + ky);
            float weight = kernel[(size_t)((ky + 1) * 3 + (kx + 1))];
            sum += sample * weight;
        }
    }

    return sum;
}

int csrc_convolve3x3_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    const float kernel[9],
    float *dst,
    size_t dst_stride
) {
    int status = validate_gray_args(src, width, height, src_stride, dst, dst_stride);
    if (status != CSRC_STATUS_OK) {
        return status;
    }
    if (kernel == NULL) {
        return CSRC_STATUS_NULL_PTR;
    }

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            dst[(y * dst_stride) + x] = apply_kernel3x3(src, width, height, src_stride, x, y, kernel);
        }
    }

    return CSRC_STATUS_OK;
}

int csrc_sobel_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    float *grad_x,
    size_t grad_x_stride,
    float *grad_y,
    size_t grad_y_stride,
    float *magnitude,
    size_t magnitude_stride
) {
    static const float kernel_x[9] = {
        -1.0f, 0.0f, 1.0f,
        -2.0f, 0.0f, 2.0f,
        -1.0f, 0.0f, 1.0f,
    };
    static const float kernel_y[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f,
    };

    int status = validate_gray_args(src, width, height, src_stride, grad_x, grad_x_stride);
    if (status != CSRC_STATUS_OK) {
        return status;
    }
    if (grad_y == NULL || magnitude == NULL || grad_y_stride < width || magnitude_stride < width) {
        return CSRC_STATUS_BAD_DIMENSIONS;
    }

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float gx = apply_kernel3x3(src, width, height, src_stride, x, y, kernel_x);
            float gy = apply_kernel3x3(src, width, height, src_stride, x, y, kernel_y);

            grad_x[(y * grad_x_stride) + x] = gx;
            grad_y[(y * grad_y_stride) + x] = gy;
            magnitude[(y * magnitude_stride) + x] = sqrtf((gx * gx) + (gy * gy));
        }
    }

    return CSRC_STATUS_OK;
}

int csrc_laplace_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    float *dst,
    size_t dst_stride
) {
    static const float kernel[9] = {
         0.0f,  1.0f,  0.0f,
         1.0f, -4.0f,  1.0f,
         0.0f,  1.0f,  0.0f,
    };

    return csrc_convolve3x3_gray_u8(src, width, height, src_stride, kernel, dst, dst_stride);
}

int csrc_harris_response_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    float k,
    float *response,
    size_t response_stride
) {
    int status = validate_gray_args(src, width, height, src_stride, response, response_stride);
    if (status != CSRC_STATUS_OK) {
        return status;
    }
    if (k <= 0.0f) {
        return CSRC_STATUS_BAD_DIMENSIONS;
    }

    size_t pixels = width * height;
    float *grad_x = (float *)malloc(sizeof(float) * pixels);
    float *grad_y = (float *)malloc(sizeof(float) * pixels);
    float *magnitude = (float *)malloc(sizeof(float) * pixels);

    if (grad_x == NULL || grad_y == NULL || magnitude == NULL) {
        free(grad_x);
        free(grad_y);
        free(magnitude);
        return CSRC_STATUS_ALLOCATION_FAILED;
    }

    status = csrc_sobel_gray_u8(src, width, height, src_stride, grad_x, width, grad_y, width, magnitude, width);
    free(magnitude);
    if (status != CSRC_STATUS_OK) {
        free(grad_x);
        free(grad_y);
        return status;
    }

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            float sxx = 0.0f;
            float syy = 0.0f;
            float sxy = 0.0f;

            for (ptrdiff_t wy = -1; wy <= 1; ++wy) {
                for (ptrdiff_t wx = -1; wx <= 1; ++wx) {
                    float gx = sample_gray_f32_clamped(grad_x, width, height, width, (ptrdiff_t)x + wx, (ptrdiff_t)y + wy);
                    float gy = sample_gray_f32_clamped(grad_y, width, height, width, (ptrdiff_t)x + wx, (ptrdiff_t)y + wy);
                    sxx += gx * gx;
                    syy += gy * gy;
                    sxy += gx * gy;
                }
            }

            {
                float det = (sxx * syy) - (sxy * sxy);
                float trace = sxx + syy;
                response[(y * response_stride) + x] = det - (k * trace * trace);
            }
        }
    }

    free(grad_x);
    free(grad_y);
    return CSRC_STATUS_OK;
}

int csrc_alpha_blend_bgra_over_bgr(
    const uint8_t *overlay_bgra,
    size_t overlay_width,
    size_t overlay_height,
    size_t overlay_stride_bytes,
    uint8_t *background_bgr,
    size_t background_width,
    size_t background_height,
    size_t background_stride_bytes,
    size_t dst_x,
    size_t dst_y
) {
    if (overlay_bgra == NULL || background_bgr == NULL) {
        return CSRC_STATUS_NULL_PTR;
    }
    if (
        overlay_width == 0 ||
        overlay_height == 0 ||
        background_width == 0 ||
        background_height == 0 ||
        overlay_stride_bytes < overlay_width * 4 ||
        background_stride_bytes < background_width * 3
    ) {
        return CSRC_STATUS_BAD_DIMENSIONS;
    }
    if (dst_x > background_width || dst_y > background_height) {
        return CSRC_STATUS_OUT_OF_BOUNDS;
    }
    if (overlay_width > background_width - dst_x || overlay_height > background_height - dst_y) {
        return CSRC_STATUS_OUT_OF_BOUNDS;
    }

    for (size_t y = 0; y < overlay_height; ++y) {
        const uint8_t *overlay_row = overlay_bgra + (y * overlay_stride_bytes);
        uint8_t *background_row = background_bgr + ((dst_y + y) * background_stride_bytes) + (dst_x * 3);

        for (size_t x = 0; x < overlay_width; ++x) {
            const uint8_t *overlay_px = overlay_row + (x * 4);
            uint8_t *background_px = background_row + (x * 3);
            float alpha = overlay_px[3] / 255.0f;
            float inv_alpha = 1.0f - alpha;

            for (size_t c = 0; c < 3; ++c) {
                float blended = (alpha * overlay_px[c]) + (inv_alpha * background_px[c]);
                background_px[c] = (uint8_t)blended;
            }
        }
    }

    return CSRC_STATUS_OK;
}
