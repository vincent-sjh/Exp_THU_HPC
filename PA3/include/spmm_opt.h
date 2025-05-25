#ifndef SPMM_OPT_H
#define SPMM_OPT_H
#include "spmm_base.h"

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) 
        : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    virtual void preprocess(float *vin, float *vout);
    virtual void run(float *vin, float *vout);
    ~SpMMOpt() {
        cudaFree(d_row_indices);
        cudaFree(d_slice_begins);
        cudaFree(d_slice_ends);
        cudaFree(d_slice_offsets);
    }

private:
    int slice_total;              // 总切片数
    int* d_row_indices;           // 设备端行索引数组
    int* d_slice_begins;          // 设备端切片起始索引数组
    int* d_slice_ends;            // 设备端切片结束索引数组
    int* d_slice_offsets;         // 设备端切片偏移数组
};
#endif