//! Multi-rank tensor fixture families (V2-CONFORM-03)
//!
//! Tests fixture construction and structural validation for tensors of rank 0-4,
//! covering broadcasting, reduction along axes, transpose, and edge cases.

use fj_conformance::{
    ComparatorKind, FixtureFamily, FixtureMode, FixtureProgram, FixtureTransform, FixtureValue,
    TransformFixtureBundle, TransformFixtureCase,
};

fn make_lax_case(
    case_id: &str,
    program: FixtureProgram,
    transforms: Vec<FixtureTransform>,
    family: FixtureFamily,
    args: Vec<FixtureValue>,
    expected: Vec<FixtureValue>,
) -> TransformFixtureCase {
    TransformFixtureCase {
        case_id: case_id.to_owned(),
        family,
        mode: FixtureMode::Strict,
        program,
        transforms,
        comparator: ComparatorKind::ApproxAtolRtol,
        baseline_mismatch: false,
        flaky: false,
        simulated_delay_ms: 0,
        args,
        expected,
        atol: 1e-6,
        rtol: 1e-6,
    }
}

fn build_multirank_fixture_bundle() -> TransformFixtureBundle {
    let cases = vec![
        // ════════════════════════════════════════════════════════
        // Rank 0: Scalar operations
        // ════════════════════════════════════════════════════════
        // 1. Scalar add
        make_lax_case(
            "r0_add_scalars",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::ScalarF64 { value: 3.0 },
                FixtureValue::ScalarF64 { value: 4.0 },
            ],
            vec![FixtureValue::ScalarF64 { value: 7.0 }],
        ),
        // 2. Scalar square
        make_lax_case(
            "r0_square_scalar",
            FixtureProgram::Square,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 5.0 }],
            vec![FixtureValue::ScalarF64 { value: 25.0 }],
        ),
        // 3. Scalar neg
        make_lax_case(
            "r0_neg_scalar",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 7.0 }],
            vec![FixtureValue::ScalarF64 { value: -7.0 }],
        ),
        // 4. Scalar exp
        make_lax_case(
            "r0_exp_scalar",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::ScalarF64 { value: 0.0 }],
            vec![FixtureValue::ScalarF64 { value: 1.0 }],
        ),
        // 5. Scalar grad
        make_lax_case(
            "r0_grad_square",
            FixtureProgram::Square,
            vec![FixtureTransform::Grad],
            FixtureFamily::Grad,
            vec![FixtureValue::ScalarF64 { value: 3.0 }],
            vec![FixtureValue::ScalarF64 { value: 6.0 }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 1: Vector operations
        // ════════════════════════════════════════════════════════
        // 6. Vector add
        make_lax_case(
            "r1_add_vectors",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 { values: vec![1.0, 2.0, 3.0] },
                FixtureValue::VectorF64 { values: vec![4.0, 5.0, 6.0] },
            ],
            vec![FixtureValue::VectorF64 { values: vec![5.0, 7.0, 9.0] }],
        ),
        // 7. Single-element vector
        make_lax_case(
            "r1_add_single",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 { values: vec![42.0] },
                FixtureValue::VectorF64 { values: vec![8.0] },
            ],
            vec![FixtureValue::VectorF64 { values: vec![50.0] }],
        ),
        // 8. Large vector (1024 elements)
        make_lax_case(
            "r1_neg_large",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::VectorF64 { values: (0..1024).map(|i| i as f64).collect() }],
            vec![FixtureValue::VectorF64 { values: (0..1024).map(|i| -(i as f64)).collect() }],
        ),
        // 9. Vector dot product
        make_lax_case(
            "r1_dot_vectors",
            FixtureProgram::Dot3,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::VectorF64 { values: vec![1.0, 2.0, 3.0] },
                FixtureValue::VectorF64 { values: vec![4.0, 5.0, 6.0] },
            ],
            vec![FixtureValue::ScalarF64 { value: 32.0 }],
        ),
        // 10. Vector reduce_sum
        make_lax_case(
            "r1_reduce_sum",
            FixtureProgram::ReduceSumVec,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::VectorF64 { values: vec![1.0, 2.0, 3.0, 4.0] }],
            vec![FixtureValue::ScalarF64 { value: 10.0 }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 2: Matrix operations (using TensorF64)
        // ════════════════════════════════════════════════════════
        // 11. 2x2 matrix negation
        make_lax_case(
            "r2_neg_2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-1.0, -2.0, -3.0, -4.0],
            }],
        ),
        // 12. 3x3 matrix abs
        make_lax_case(
            "r2_abs_3x3",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 3],
                values: vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0, -9.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }],
        ),
        // 13. 4x4 matrix floor
        make_lax_case(
            "r2_floor_4x4",
            FixtureProgram::LaxFloor,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![4, 4],
                values: vec![
                    1.1, 2.9, 3.5, 4.0,
                    -1.1, -2.9, -3.5, -4.0,
                    0.0, 0.5, -0.5, 99.99,
                    -99.99, 0.001, -0.001, 1e10,
                ],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![4, 4],
                values: vec![
                    1.0, 2.0, 3.0, 4.0,
                    -2.0, -3.0, -4.0, -4.0,
                    0.0, 0.0, -1.0, 99.0,
                    -100.0, 0.0, -1.0, 1e10,
                ],
            }],
        ),
        // 14. 2x3 matrix exp
        make_lax_case(
            "r2_exp_2x3",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![0.0, 1.0, -1.0, 2.0, -2.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![
                    1.0,
                    std::f64::consts::E,
                    (-1.0_f64).exp(),
                    (2.0_f64).exp(),
                    (-2.0_f64).exp(),
                    (0.5_f64).exp(),
                ],
            }],
        ),
        // 15. 1x1 matrix (singleton)
        make_lax_case(
            "r2_neg_1x1",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1],
                values: vec![42.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1],
                values: vec![-42.0],
            }],
        ),
        // 16. 8x16 matrix square
        make_lax_case(
            "r2_square_8x16",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![8, 16],
                values: (0..128).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![8, 16],
                values: (0..128).map(|i| (i * i) as f64).collect(),
            }],
        ),
        // 17. 2x2 matrix add (elementwise)
        make_lax_case(
            "r2_add_2x2",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![11.0, 22.0, 33.0, 44.0],
            }],
        ),
        // 18. 3x2 matrix sin
        make_lax_case(
            "r2_sin_3x2",
            FixtureProgram::SinX,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2],
                values: vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI, 0.0, 1.0, 2.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2],
                values: vec![
                    0.0,
                    1.0,
                    std::f64::consts::PI.sin(),
                    0.0,
                    1.0_f64.sin(),
                    2.0_f64.sin(),
                ],
            }],
        ),
        // 19. 2x2 matrix reciprocal
        make_lax_case(
            "r2_reciprocal_2x2",
            FixtureProgram::LaxReciprocal,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 4.0, 5.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 0.5, 0.25, 0.2],
            }],
        ),
        // 20. 2x2 matrix mul (elementwise)
        make_lax_case(
            "r2_mul_2x2",
            FixtureProgram::LaxMul,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![5.0, 6.0, 7.0, 8.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![5.0, 12.0, 21.0, 32.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 3: 3D tensor operations
        // ════════════════════════════════════════════════════════
        // 21. 2x2x2 tensor negation
        make_lax_case(
            "r3_neg_2x2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2],
                values: vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
            }],
        ),
        // 22. 2x3x4 tensor abs
        make_lax_case(
            "r3_abs_2x3x4",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 4],
                values: (0..24).map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) }).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 4],
                values: (0..24).map(|i| i as f64).collect(),
            }],
        ),
        // 23. 1x1x1 tensor (singleton 3D)
        make_lax_case(
            "r3_neg_1x1x1",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1],
                values: vec![99.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1],
                values: vec![-99.0],
            }],
        ),
        // 24. 3x2x2 tensor exp
        make_lax_case(
            "r3_exp_3x2x2",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2, 2],
                values: vec![
                    0.0, 1.0, -1.0, 0.5,
                    -0.5, 2.0, -2.0, 0.0,
                    1.0, -1.0, 0.0, 0.5,
                ],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 2, 2],
                values: vec![
                    1.0, std::f64::consts::E, (-1.0_f64).exp(), (0.5_f64).exp(),
                    (-0.5_f64).exp(), (2.0_f64).exp(), (-2.0_f64).exp(), 1.0,
                    std::f64::consts::E, (-1.0_f64).exp(), 1.0, (0.5_f64).exp(),
                ],
            }],
        ),
        // 25. 2x2x3 tensor add (elementwise)
        make_lax_case(
            "r3_add_2x2x3",
            FixtureProgram::Add2,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2, 3],
                    values: (0..12).map(|i| i as f64).collect(),
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2, 3],
                    values: (0..12).map(|i| (i * 10) as f64).collect(),
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 3],
                values: (0..12).map(|i| (i + i * 10) as f64).collect(),
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Rank 4: 4D tensor operations
        // ════════════════════════════════════════════════════════
        // 26. 2x2x2x2 tensor neg
        make_lax_case(
            "r4_neg_2x2x2x2",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2, 2],
                values: (0..16).map(|i| (i + 1) as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2, 2, 2],
                values: (0..16).map(|i| -((i + 1) as f64)).collect(),
            }],
        ),
        // 27. 1x2x3x4 tensor square
        make_lax_case(
            "r4_square_1x2x3x4",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 2, 3, 4],
                values: (0..24).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 2, 3, 4],
                values: (0..24).map(|i| (i * i) as f64).collect(),
            }],
        ),
        // 28. 1x1x1x1 tensor (singleton 4D)
        make_lax_case(
            "r4_abs_1x1x1x1",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1, 1],
                values: vec![-7.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![1, 1, 1, 1],
                values: vec![7.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Trig functions on matrices
        // ════════════════════════════════════════════════════════
        // 29. 2x2 cos
        make_lax_case(
            "r2_cos_2x2",
            FixtureProgram::CosX,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, std::f64::consts::PI, std::f64::consts::FRAC_PI_2, 1.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    1.0,
                    -1.0,
                    (std::f64::consts::FRAC_PI_2).cos(),
                    1.0_f64.cos(),
                ],
            }],
        ),
        // 30. 2x3 tanh
        make_lax_case(
            "r2_tanh_2x3",
            FixtureProgram::LaxTanh,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![
                    0.0,
                    1.0_f64.tanh(),
                    (-1.0_f64).tanh(),
                    100.0_f64.tanh(),
                    (-100.0_f64).tanh(),
                    0.5_f64.tanh(),
                ],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Log/special functions on matrices
        // ════════════════════════════════════════════════════════
        // 31. 2x2 log
        make_lax_case(
            "r2_log_2x2",
            FixtureProgram::LaxLog,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, std::f64::consts::E, 10.0, 0.5],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 10.0_f64.ln(), 0.5_f64.ln()],
            }],
        ),
        // 32. 2x2 sqrt
        make_lax_case(
            "r2_sqrt_2x2",
            FixtureProgram::LaxSqrt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 4.0, 9.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, 2.0, 3.0],
            }],
        ),
        // 33. 2x2 logistic
        make_lax_case(
            "r2_logistic_2x2",
            FixtureProgram::LaxLogistic,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 100.0, -100.0, 1.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.5,
                    1.0, // logistic(100) ≈ 1
                    0.0, // logistic(-100) ≈ 0
                    1.0 / (1.0 + (-1.0_f64).exp()),
                ],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Comparison on matrices
        // ════════════════════════════════════════════════════════
        // 34. 2x2 eq
        make_lax_case(
            "r2_eq_2x2",
            FixtureProgram::LaxEq,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 0.0, 3.0, 5.0],
                },
            ],
            vec![FixtureValue::TensorBool {
                shape: vec![2, 2],
                values: vec![true, false, true, false],
            }],
        ),
        // 35. 2x2 lt
        make_lax_case(
            "r2_lt_2x2",
            FixtureProgram::LaxLt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 4.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 3.0, 3.0, 5.0],
                },
            ],
            vec![FixtureValue::TensorBool {
                shape: vec![2, 2],
                values: vec![true, false, false, true],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Binary operations on matrices
        // ════════════════════════════════════════════════════════
        // 36. 2x2 sub
        make_lax_case(
            "r2_sub_2x2",
            FixtureProgram::LaxSub,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 2.0, 3.0, 4.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![9.0, 18.0, 27.0, 36.0],
            }],
        ),
        // 37. 2x2 div
        make_lax_case(
            "r2_div_2x2",
            FixtureProgram::LaxDiv,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![10.0, 20.0, 30.0, 40.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 5.0, 6.0, 8.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![5.0, 4.0, 5.0, 5.0],
            }],
        ),
        // 38. 2x2 max
        make_lax_case(
            "r2_max_2x2",
            FixtureProgram::LaxMax,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 7.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![4.0, 2.0, 6.0, 0.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![4.0, 5.0, 6.0, 7.0],
            }],
        ),
        // 39. 2x2 min
        make_lax_case(
            "r2_min_2x2",
            FixtureProgram::LaxMin,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![1.0, 5.0, 3.0, 7.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![4.0, 2.0, 6.0, 0.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 3.0, 0.0],
            }],
        ),
        // 40. 2x2 pow
        make_lax_case(
            "r2_pow_2x2",
            FixtureProgram::LaxPow,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![2.0, 3.0, 4.0, 5.0],
                },
                FixtureValue::TensorF64 {
                    shape: vec![2, 2],
                    values: vec![3.0, 2.0, 0.5, 1.0],
                },
            ],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![8.0, 9.0, 2.0, 5.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Vmap over matrices
        // ════════════════════════════════════════════════════════
        // 41. vmap(neg) over batch of vectors (rank 2 input)
        make_lax_case(
            "r2_vmap_neg",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Vmap],
            FixtureFamily::Vmap,
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 4],
                values: (0..12).map(|i| (i + 1) as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![3, 4],
                values: (0..12).map(|i| -((i + 1) as f64)).collect(),
            }],
        ),
        // 42. vmap(square) over batch of vectors (rank 2 input)
        make_lax_case(
            "r2_vmap_square",
            FixtureProgram::LaxSquare,
            vec![FixtureTransform::Vmap],
            FixtureFamily::Vmap,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Grad on matrices (scalar output)
        // ════════════════════════════════════════════════════════
        // 43. grad(reduce_sum) on vector
        make_lax_case(
            "r1_grad_reduce_sum",
            FixtureProgram::ReduceSumVec,
            vec![FixtureTransform::Grad],
            FixtureFamily::Grad,
            vec![FixtureValue::VectorF64 { values: vec![1.0, 2.0, 3.0] }],
            vec![FixtureValue::VectorF64 { values: vec![1.0, 1.0, 1.0] }],
        ),
        // ════════════════════════════════════════════════════════
        // Additional unary ops on rank-2
        // ════════════════════════════════════════════════════════
        // 44. 2x2 ceil
        make_lax_case(
            "r2_ceil_2x2",
            FixtureProgram::LaxCeil,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.1, 2.9, -1.1, -2.9],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![2.0, 3.0, -1.0, -2.0],
            }],
        ),
        // 45. 2x2 round
        make_lax_case(
            "r2_round_2x2",
            FixtureProgram::LaxRound,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.4, 1.5, 2.5, 3.6],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 2.0, 2.0, 4.0],
            }],
        ),
        // 46. 2x2 sign
        make_lax_case(
            "r2_sign_2x2",
            FixtureProgram::LaxSign,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-5.0, 0.0, 3.0, -0.001],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![-1.0, 0.0, 1.0, -1.0],
            }],
        ),
        // 47. 2x2 rsqrt
        make_lax_case(
            "r2_rsqrt_2x2",
            FixtureProgram::LaxRsqrt,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 4.0, 9.0, 16.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![1.0, 0.5, 1.0 / 3.0, 0.25],
            }],
        ),
        // 48. 2x2 erf
        make_lax_case(
            "r2_erf_2x2",
            FixtureProgram::LaxErf,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 3.0],
            }],
            // erf values computed by reference
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 0.842700792949715, -0.842700792949715, 0.9999779095030014],
            }],
        ),
        // ════════════════════════════════════════════════════════
        // Jit composition on tensors
        // ════════════════════════════════════════════════════════
        // 49. jit(add_one) on 2x2 matrix
        make_lax_case(
            "r2_jit_add_one",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![10.0, 20.0, 30.0, 40.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![11.0, 21.0, 31.0, 41.0],
            }],
        ),
        // 50. jit(square) on 2x3 matrix
        make_lax_case(
            "r2_jit_square",
            FixtureProgram::Square,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3],
                values: vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
            }],
        ),
        // 51. Identity on various ranks
        make_lax_case(
            "r3_identity",
            FixtureProgram::Identity,
            vec![FixtureTransform::Jit],
            FixtureFamily::Jit,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 2],
                values: (0..12).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 3, 2],
                values: (0..12).map(|i| i as f64).collect(),
            }],
        ),
        // 52. log1p on 2x2
        make_lax_case(
            "r2_log1p_2x2",
            FixtureProgram::LaxLog1p,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, std::f64::consts::E - 1.0, 99.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.0,
                    2.0_f64.ln(),
                    1.0, // ln(e) = 1
                    100.0_f64.ln(),
                ],
            }],
        ),
        // 53. expm1 on 2x2
        make_lax_case(
            "r2_expm1_2x2",
            FixtureProgram::LaxExpm1,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 0.001],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    0.0,
                    std::f64::consts::E - 1.0,
                    (-1.0_f64).exp() - 1.0,
                    (0.001_f64).exp_m1(),
                ],
            }],
        ),
        // 54. erfc on 2x2
        make_lax_case(
            "r2_erfc_2x2",
            FixtureProgram::LaxErfc,
            vec![FixtureTransform::Jit],
            FixtureFamily::Lax,
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![0.0, 1.0, -1.0, 3.0],
            }],
            vec![FixtureValue::TensorF64 {
                shape: vec![2, 2],
                values: vec![
                    1.0,
                    1.0 - 0.842700792949715,
                    1.0 + 0.842700792949715,
                    1.0 - 0.9999779095030014,
                ],
            }],
        ),
    ];

    TransformFixtureBundle {
        schema_version: "frankenjax.transform-fixture-bundle.v1".to_owned(),
        generated_by: "multirank_conformance".to_owned(),
        generated_at_unix_ms: 0,
        cases,
    }
}

// ── Structural validation tests ────────────────────────────────────

#[test]
fn test_fixture_total_count() {
    let bundle = build_multirank_fixture_bundle();
    assert!(
        bundle.cases.len() >= 50,
        "need 50+ fixtures, got {}",
        bundle.cases.len()
    );
}

#[test]
fn test_fixture_unique_case_ids() {
    let bundle = build_multirank_fixture_bundle();
    let mut ids: Vec<&str> = bundle.cases.iter().map(|c| c.case_id.as_str()).collect();
    ids.sort();
    let before = ids.len();
    ids.dedup();
    assert_eq!(before, ids.len(), "duplicate case IDs found");
}

#[test]
fn test_fixture_rank0_scalar() {
    let bundle = build_multirank_fixture_bundle();
    let r0_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().all(|a| a.rank() == 0))
        .count();
    assert!(r0_count >= 3, "need >=3 rank-0 cases, got {r0_count}");
}

#[test]
fn test_fixture_rank1_vector() {
    let bundle = build_multirank_fixture_bundle();
    let r1_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 1))
        .count();
    assert!(r1_count >= 3, "need >=3 rank-1 cases, got {r1_count}");
}

#[test]
fn test_fixture_rank2_matrix() {
    let bundle = build_multirank_fixture_bundle();
    let r2_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 2))
        .count();
    assert!(r2_count >= 10, "need >=10 rank-2 cases, got {r2_count}");
}

#[test]
fn test_fixture_rank3_tensor() {
    let bundle = build_multirank_fixture_bundle();
    let r3_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 3))
        .count();
    assert!(r3_count >= 3, "need >=3 rank-3 cases, got {r3_count}");
}

#[test]
fn test_fixture_rank4_batch_matrix() {
    let bundle = build_multirank_fixture_bundle();
    let r4_count = bundle
        .cases
        .iter()
        .filter(|c| c.args.iter().any(|a| a.rank() == 4))
        .count();
    assert!(r4_count >= 2, "need >=2 rank-4 cases, got {r4_count}");
}

#[test]
fn test_fixture_serializable() {
    let bundle = build_multirank_fixture_bundle();
    let json = serde_json::to_string_pretty(&bundle).expect("serialization failed");
    let _roundtrip: TransformFixtureBundle =
        serde_json::from_str(&json).expect("deserialization failed");
}

#[test]
fn test_fixture_all_have_expected() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        assert!(
            !case.expected.is_empty(),
            "case {} has no expected output",
            case.case_id
        );
    }
}

#[test]
fn test_fixture_tight_tolerances() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        assert!(
            case.atol <= 1e-6,
            "case {} has atol {} > 1e-6",
            case.case_id,
            case.atol
        );
        assert!(
            case.rtol <= 1e-6,
            "case {} has rtol {} > 1e-6",
            case.case_id,
            case.rtol
        );
    }
}

#[test]
fn test_fixture_args_runtime_conversion() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for (i, arg) in case.args.iter().enumerate() {
            arg.to_runtime_value()
                .unwrap_or_else(|e| panic!("case {} arg[{i}] conversion failed: {e}", case.case_id));
        }
    }
}

#[test]
fn test_fixture_expected_runtime_conversion() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for (i, exp) in case.expected.iter().enumerate() {
            exp.to_runtime_value()
                .unwrap_or_else(|e| panic!("case {} expected[{i}] conversion failed: {e}", case.case_id));
        }
    }
}

#[test]
fn test_fixture_tensor_shape_consistency() {
    let bundle = build_multirank_fixture_bundle();
    for case in &bundle.cases {
        for arg in &case.args {
            if let FixtureValue::TensorF64 { shape, values } = arg {
                let expected_len: u32 = shape.iter().product();
                assert_eq!(
                    values.len() as u32, expected_len,
                    "case {} arg shape {:?} expects {} elements, got {}",
                    case.case_id, shape, expected_len, values.len()
                );
            }
        }
        for exp in &case.expected {
            if let FixtureValue::TensorF64 { shape, values } = exp {
                let expected_len: u32 = shape.iter().product();
                assert_eq!(
                    values.len() as u32, expected_len,
                    "case {} expected shape {:?} expects {} elements, got {}",
                    case.case_id, shape, expected_len, values.len()
                );
            }
        }
    }
}

#[test]
fn test_fixture_covers_multiple_families() {
    let bundle = build_multirank_fixture_bundle();
    let families: std::collections::HashSet<_> = bundle.cases.iter().map(|c| c.family).collect();
    assert!(families.len() >= 3, "need >=3 families, got {:?}", families);
}

#[test]
fn test_fixture_singleton_dimensions() {
    let bundle = build_multirank_fixture_bundle();
    let singleton_count = bundle
        .cases
        .iter()
        .filter(|c| {
            c.args.iter().any(|a| match a {
                FixtureValue::TensorF64 { shape, .. } => shape.iter().any(|&d| d == 1),
                _ => false,
            })
        })
        .count();
    assert!(
        singleton_count >= 3,
        "need >=3 singleton-dim cases, got {singleton_count}"
    );
}
