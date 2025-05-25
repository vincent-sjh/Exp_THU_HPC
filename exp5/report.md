# 小作业五 自动向量化与基于 intrinsic 的手动向量化

宋建昊 2022010853

## 测试结果

| 版本      | 运行时间(us) |
| --------- | ------------ |
| baseline  | 4438         |
| auto SIMD | 526          |
| intrinsic | 527          |

## 实现代码

```cpp
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 a_vec = _mm256_load_ps(&a[i]);
        __m256 b_vec = _mm256_load_ps(&b[i]);
        __m256 c_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_store_ps(&c[i], c_vec);
    }
}
```
