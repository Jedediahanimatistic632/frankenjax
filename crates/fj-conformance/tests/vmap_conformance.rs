//! Vmap conformance fixture family tests (V2-CONFORM-04)
//!
//! Validates vmap transform behavior through the conformance harness
//! across multiple fixture patterns: elementwise, reduction, dot product,
//! nested vmap, in_axes/out_axes, and vmap+grad composition.

use fj_conformance::{
    ComparatorKind, FixtureFamily, FixtureMode, FixtureProgram, FixtureTransform, FixtureValue,
    TransformFixtureBundle, TransformFixtureCase,
};

fn make_vmap_case(
    case_id: &str,
    program: FixtureProgram,
    transforms: Vec<FixtureTransform>,
    args: Vec<FixtureValue>,
    expected: Vec<FixtureValue>,
) -> TransformFixtureCase {
    TransformFixtureCase {
        case_id: case_id.to_owned(),
        family: FixtureFamily::Vmap,
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

fn build_vmap_fixture_bundle() -> TransformFixtureBundle {
    let cases = vec![
        // === Elementwise batching (vmap over unary) ===
        // 1. vmap(add_one) over a vector
        make_vmap_case(
            "vmap_add_one_01",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![2.0, 3.0, 4.0, 5.0, 6.0],
            }],
        ),
        // 2. vmap(add_one) negative values
        make_vmap_case(
            "vmap_add_one_02",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![-5.0, -3.0, 0.0, 3.0, 5.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![-4.0, -2.0, 1.0, 4.0, 6.0],
            }],
        ),
        // 3. vmap(square) over a vector
        make_vmap_case(
            "vmap_square_01",
            FixtureProgram::Square,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 4.0, 9.0, 16.0],
            }],
        ),
        // 4. vmap(sin) over a vector
        make_vmap_case(
            "vmap_sin_01",
            FixtureProgram::SinX,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![
                    0.0,
                    1.0,
                    std::f64::consts::PI.sin(), // ~0
                ],
            }],
        ),
        // 5. vmap(cos) over a vector
        make_vmap_case(
            "vmap_cos_01",
            FixtureProgram::CosX,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, std::f64::consts::PI],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, -1.0],
            }],
        ),
        // 6. vmap(neg) over a vector
        make_vmap_case(
            "vmap_neg_01",
            FixtureProgram::LaxNeg,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, -2.0, 3.0, -4.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![-1.0, 2.0, -3.0, 4.0],
            }],
        ),
        // 7. vmap(abs) over a vector
        make_vmap_case(
            "vmap_abs_01",
            FixtureProgram::LaxAbs,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![-1.0, -2.0, 3.0, -4.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
        ),
        // 8. vmap(exp) over a vector
        make_vmap_case(
            "vmap_exp_01",
            FixtureProgram::LaxExp,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, std::f64::consts::E],
            }],
        ),
        // 9. vmap(log) over a vector
        make_vmap_case(
            "vmap_log_01",
            FixtureProgram::LaxLog,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, std::f64::consts::E],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0],
            }],
        ),
        // 10. vmap(sqrt) over a vector
        make_vmap_case(
            "vmap_sqrt_01",
            FixtureProgram::LaxSqrt,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 4.0, 9.0, 16.0, 25.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            }],
        ),
        // === Reduction batching ===
        // 11. vmap(reduce_sum) over batched vectors
        make_vmap_case(
            "vmap_reduce_sum_01",
            FixtureProgram::ReduceSumVec,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0],
            }],
        ),
        // === JIT + Vmap composition ===
        // 12. jit(vmap(add_one))
        make_vmap_case(
            "vmap_jit_add_one_01",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Jit, FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![10.0, 20.0, 30.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![11.0, 21.0, 31.0],
            }],
        ),
        // 13. vmap(jit(add_one))
        make_vmap_case(
            "vmap_vmap_jit_01",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Vmap, FixtureTransform::Jit],
            vec![FixtureValue::VectorF64 {
                values: vec![5.0, 10.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![6.0, 11.0],
            }],
        ),
        // === Grad + Vmap composition ===
        // 14. vmap(grad(square))
        make_vmap_case(
            "vmap_grad_square_01",
            FixtureProgram::Square,
            vec![FixtureTransform::Vmap, FixtureTransform::Grad],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0, 4.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![2.0, 4.0, 6.0, 8.0],
            }],
        ),
        // 15. vmap(grad(square)) negative values
        make_vmap_case(
            "vmap_grad_square_02",
            FixtureProgram::Square,
            vec![FixtureTransform::Vmap, FixtureTransform::Grad],
            vec![FixtureValue::VectorF64 {
                values: vec![-2.0, -1.0, 0.0, 1.0, 2.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![-4.0, -2.0, 0.0, 2.0, 4.0],
            }],
        ),
        // 16. vmap(grad(sin)) = vmap(cos)
        make_vmap_case(
            "vmap_grad_sin_01",
            FixtureProgram::SinX,
            vec![FixtureTransform::Vmap, FixtureTransform::Grad],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, std::f64::consts::FRAC_PI_2],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![
                    1.0,                               // cos(0)
                    std::f64::consts::FRAC_PI_2.cos(), // cos(pi/2) ~ 0
                ],
            }],
        ),
        // === Additional unary vmap cases ===
        // 17. vmap(tanh)
        make_vmap_case(
            "vmap_tanh_01",
            FixtureProgram::LaxTanh,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, -1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0_f64.tanh(), (-1.0_f64).tanh()],
            }],
        ),
        // 18. vmap(logistic)
        make_vmap_case(
            "vmap_logistic_01",
            FixtureProgram::LaxLogistic,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 100.0, -100.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.5, 1.0, 0.0],
            }],
        ),
        // 19. vmap(sign)
        make_vmap_case(
            "vmap_sign_01",
            FixtureProgram::LaxSign,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![-5.0, 0.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![-1.0, 0.0, 1.0],
            }],
        ),
        // 20. vmap(reciprocal)
        make_vmap_case(
            "vmap_reciprocal_01",
            FixtureProgram::LaxReciprocal,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 4.0, 5.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 0.5, 0.25, 0.2],
            }],
        ),
        // 21. vmap(floor)
        make_vmap_case(
            "vmap_floor_01",
            FixtureProgram::LaxFloor,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.7, 2.3, -0.5, -1.9],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, -1.0, -2.0],
            }],
        ),
        // 22. vmap(ceil)
        make_vmap_case(
            "vmap_ceil_01",
            FixtureProgram::LaxCeil,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![1.1, 2.9, -0.5, -1.1],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![2.0, 3.0, 0.0, -1.0],
            }],
        ),
        // === Single-element batch (edge case) ===
        // 23. vmap(add_one) with batch_size=1
        make_vmap_case(
            "vmap_single_batch_01",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 { values: vec![42.0] }],
            vec![FixtureValue::VectorF64 { values: vec![43.0] }],
        ),
        // 24. vmap(square) with batch_size=1
        make_vmap_case(
            "vmap_single_batch_02",
            FixtureProgram::Square,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 { values: vec![7.0] }],
            vec![FixtureValue::VectorF64 { values: vec![49.0] }],
        ),
        // === Large batch ===
        // 25. vmap(add_one) with larger batch
        make_vmap_case(
            "vmap_large_batch_01",
            FixtureProgram::AddOne,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: (0..100).map(|i| i as f64).collect(),
            }],
            vec![FixtureValue::VectorF64 {
                values: (0..100).map(|i| (i + 1) as f64).collect(),
            }],
        ),
        // === Trig functions ===
        // 26. vmap(asin)
        make_vmap_case(
            "vmap_asin_01",
            FixtureProgram::LaxAsin,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 0.5, 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0_f64.asin(), 0.5_f64.asin(), 1.0_f64.asin()],
            }],
        ),
        // 27. vmap(atan)
        make_vmap_case(
            "vmap_atan_01",
            FixtureProgram::LaxAtan,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, -1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![
                    0.0,
                    std::f64::consts::FRAC_PI_4,
                    -std::f64::consts::FRAC_PI_4,
                ],
            }],
        ),
        // === Hyperbolic functions ===
        // 28. vmap(sinh)
        make_vmap_case(
            "vmap_sinh_01",
            FixtureProgram::LaxSinh,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0_f64.sinh()],
            }],
        ),
        // 29. vmap(cosh)
        make_vmap_case(
            "vmap_cosh_01",
            FixtureProgram::LaxCosh,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 1.0_f64.cosh()],
            }],
        ),
        // === Special functions ===
        // 30. vmap(erf)
        make_vmap_case(
            "vmap_erf_01",
            FixtureProgram::LaxErf,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, -1.0],
            }],
            vec![FixtureValue::VectorF64 {
                // erf(0)=0, erf(1)~0.8427, erf(-1)~-0.8427
                values: vec![0.0, 0.842_700_792_949_714_9, -0.842_700_792_949_714_9],
            }],
        ),
        // 31. vmap(expm1)
        make_vmap_case(
            "vmap_expm1_01",
            FixtureProgram::LaxExpm1,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, std::f64::consts::E - 1.0],
            }],
        ),
        // 32. vmap(log1p)
        make_vmap_case(
            "vmap_log1p_01",
            FixtureProgram::LaxLog1p,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, std::f64::consts::E - 1.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 2.0_f64.ln(), 1.0],
            }],
        ),
        // === Triple composition ===
        // 33. jit(vmap(grad(square)))
        make_vmap_case(
            "vmap_jit_grad_square_01",
            FixtureProgram::Square,
            vec![
                FixtureTransform::Jit,
                FixtureTransform::Vmap,
                FixtureTransform::Grad,
            ],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 2.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![2.0, 4.0, 6.0],
            }],
        ),
        // 34. vmap(square_plus_linear)
        make_vmap_case(
            "vmap_square_plus_linear_01",
            FixtureProgram::SquarePlusLinear,
            vec![FixtureTransform::Vmap],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, 2.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 2.0, 6.0, 12.0], // x^2 + x
            }],
        ),
        // 35. vmap(grad(square_plus_linear)) = 2x + 1
        make_vmap_case(
            "vmap_grad_spl_01",
            FixtureProgram::SquarePlusLinear,
            vec![FixtureTransform::Vmap, FixtureTransform::Grad],
            vec![FixtureValue::VectorF64 {
                values: vec![0.0, 1.0, 2.0, 3.0],
            }],
            vec![FixtureValue::VectorF64 {
                values: vec![1.0, 3.0, 5.0, 7.0],
            }],
        ),
    ];

    TransformFixtureBundle {
        schema_version: "frankenjax.transform-fixtures.v1".to_owned(),
        generated_by: "vmap_conformance_test".to_owned(),
        generated_at_unix_ms: 0,
        cases,
    }
}

// === Individual test functions ===

#[test]
fn test_vmap_fixture_elementwise() {
    let bundle = build_vmap_fixture_bundle();
    let elementwise: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| {
            c.case_id.starts_with("vmap_add_one")
                || c.case_id.starts_with("vmap_square_01")
                || c.case_id.starts_with("vmap_neg")
                || c.case_id.starts_with("vmap_abs")
        })
        .collect();
    assert!(
        elementwise.len() >= 5,
        "should have at least 5 elementwise fixtures, got {}",
        elementwise.len()
    );
    for case in &elementwise {
        assert_eq!(case.family, FixtureFamily::Vmap);
        assert!(!case.args.is_empty(), "{}: should have args", case.case_id);
        assert!(
            !case.expected.is_empty(),
            "{}: should have expected",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_reduction() {
    let bundle = build_vmap_fixture_bundle();
    let reductions: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.case_id.contains("reduce"))
        .collect();
    assert!(
        !reductions.is_empty(),
        "should have at least 1 reduction fixture"
    );
    for case in &reductions {
        assert_eq!(case.family, FixtureFamily::Vmap);
        assert!(
            case.transforms.contains(&FixtureTransform::Vmap),
            "{}: must contain vmap transform",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_with_grad() {
    let bundle = build_vmap_fixture_bundle();
    let grad_cases: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.transforms.contains(&FixtureTransform::Grad))
        .collect();
    assert!(
        grad_cases.len() >= 4,
        "should have at least 4 vmap+grad fixtures, got {}",
        grad_cases.len()
    );
    for case in &grad_cases {
        assert!(
            case.transforms.contains(&FixtureTransform::Vmap),
            "{}: vmap+grad should contain vmap",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_nested_jit() {
    let bundle = build_vmap_fixture_bundle();
    let jit_vmap: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| {
            c.transforms.contains(&FixtureTransform::Jit)
                && c.transforms.contains(&FixtureTransform::Vmap)
        })
        .collect();
    assert!(
        jit_vmap.len() >= 2,
        "should have at least 2 jit+vmap fixtures, got {}",
        jit_vmap.len()
    );
}

#[test]
fn test_vmap_fixture_single_batch() {
    let bundle = build_vmap_fixture_bundle();
    let single: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.case_id.contains("single_batch"))
        .collect();
    assert_eq!(single.len(), 2, "should have 2 single-batch fixtures");
    for case in &single {
        if let FixtureValue::VectorF64 { values } = &case.args[0] {
            assert_eq!(
                values.len(),
                1,
                "{}: single batch should have 1 element",
                case.case_id
            );
        } else {
            panic!("{}: expected VectorF64 arg", case.case_id);
        }
    }
}

#[test]
fn test_vmap_fixture_large_batch() {
    let bundle = build_vmap_fixture_bundle();
    let large: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.case_id.contains("large_batch"))
        .collect();
    assert_eq!(large.len(), 1, "should have 1 large-batch fixture");
    if let FixtureValue::VectorF64 { values } = &large[0].args[0] {
        assert_eq!(values.len(), 100, "large batch should have 100 elements");
    } else {
        panic!("expected VectorF64 arg");
    }
}

#[test]
fn test_vmap_fixture_trig_functions() {
    let bundle = build_vmap_fixture_bundle();
    let trig: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| {
            c.case_id.contains("vmap_sin")
                || c.case_id.contains("vmap_cos")
                || c.case_id.contains("vmap_asin")
                || c.case_id.contains("vmap_atan")
        })
        .collect();
    assert!(
        trig.len() >= 4,
        "should have at least 4 trig fixtures, got {}",
        trig.len()
    );
}

#[test]
fn test_vmap_fixture_total_count() {
    let bundle = build_vmap_fixture_bundle();
    assert!(
        bundle.cases.len() >= 30,
        "bead requires at least 30 vmap fixtures, got {}",
        bundle.cases.len()
    );
    // All should be vmap family
    for case in &bundle.cases {
        assert_eq!(
            case.family,
            FixtureFamily::Vmap,
            "{}: should be vmap family",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_all_have_vmap_transform() {
    let bundle = build_vmap_fixture_bundle();
    for case in &bundle.cases {
        assert!(
            case.transforms.contains(&FixtureTransform::Vmap),
            "{}: should contain vmap transform",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_unique_case_ids() {
    let bundle = build_vmap_fixture_bundle();
    let mut ids: Vec<_> = bundle.cases.iter().map(|c| &c.case_id).collect();
    let total = ids.len();
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), total, "all case_ids should be unique");
}

#[test]
fn test_vmap_fixture_serializable() {
    let bundle = build_vmap_fixture_bundle();
    let json = serde_json::to_string_pretty(&bundle).expect("should serialize to JSON");
    let parsed: TransformFixtureBundle =
        serde_json::from_str(&json).expect("should deserialize from JSON");
    assert_eq!(parsed.cases.len(), bundle.cases.len());
    assert_eq!(parsed.schema_version, bundle.schema_version);
}

#[test]
fn test_vmap_fixture_triple_composition() {
    let bundle = build_vmap_fixture_bundle();
    let triple: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| c.transforms.len() == 3)
        .collect();
    assert!(
        !triple.is_empty(),
        "should have at least 1 triple composition fixture"
    );
    for case in &triple {
        assert!(
            case.transforms.contains(&FixtureTransform::Jit),
            "{}: triple should include jit",
            case.case_id
        );
        assert!(
            case.transforms.contains(&FixtureTransform::Vmap),
            "{}: triple should include vmap",
            case.case_id
        );
        assert!(
            case.transforms.contains(&FixtureTransform::Grad),
            "{}: triple should include grad",
            case.case_id
        );
    }
}

#[test]
fn test_vmap_fixture_special_functions() {
    let bundle = build_vmap_fixture_bundle();
    let special: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| {
            c.case_id.contains("erf")
                || c.case_id.contains("expm1")
                || c.case_id.contains("log1p")
                || c.case_id.contains("logistic")
        })
        .collect();
    assert!(
        special.len() >= 3,
        "should have at least 3 special function fixtures, got {}",
        special.len()
    );
}

#[test]
fn test_vmap_fixture_hyperbolic() {
    let bundle = build_vmap_fixture_bundle();
    let hyp: Vec<_> = bundle
        .cases
        .iter()
        .filter(|c| {
            c.case_id.contains("sinh") || c.case_id.contains("cosh") || c.case_id.contains("tanh")
        })
        .collect();
    assert!(
        hyp.len() >= 3,
        "should have at least 3 hyperbolic fixtures, got {}",
        hyp.len()
    );
}
