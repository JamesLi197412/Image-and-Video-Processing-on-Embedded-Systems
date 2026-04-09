#ifndef CSRC_IMAGE_OPS_H
#define CSRC_IMAGE_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    CSRC_STATUS_OK = 0,
    CSRC_STATUS_NULL_PTR = -1,
    CSRC_STATUS_BAD_DIMENSIONS = -2,
    CSRC_STATUS_OUT_OF_BOUNDS = -3,
    CSRC_STATUS_ALLOCATION_FAILED = -4
};

int csrc_convolve3x3_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    const float kernel[9],
    float *dst,
    size_t dst_stride
);

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
);

int csrc_laplace_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    float *dst,
    size_t dst_stride
);

int csrc_harris_response_gray_u8(
    const uint8_t *src,
    size_t width,
    size_t height,
    size_t src_stride,
    float k,
    float *response,
    size_t response_stride
);

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
);

#ifdef __cplusplus
}
#endif

#endif
