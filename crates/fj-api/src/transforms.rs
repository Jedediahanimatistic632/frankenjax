#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use fj_core::{CompatibilityMode, Jaxpr, TraceTransformLedger, Transform, Value};
use fj_dispatch::{DispatchRequest, dispatch};

use crate::errors::ApiError;

#[derive(Debug, Clone, PartialEq)]
pub struct JitWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VmapWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
    in_axes: Option<String>,
    out_axes: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValueAndGradWrapped {
    jaxpr: Jaxpr,
    backend: String,
    mode: CompatibilityMode,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ComposedTransform {
    jaxpr: Jaxpr,
    transforms: Vec<Transform>,
    backend: String,
    mode: CompatibilityMode,
}

#[must_use]
pub fn jit(jaxpr: Jaxpr) -> JitWrapped {
    JitWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

#[must_use]
pub fn grad(jaxpr: Jaxpr) -> GradWrapped {
    GradWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

#[must_use]
pub fn vmap(jaxpr: Jaxpr) -> VmapWrapped {
    VmapWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
        in_axes: None,
        out_axes: None,
    }
}

#[must_use]
pub fn value_and_grad(jaxpr: Jaxpr) -> ValueAndGradWrapped {
    ValueAndGradWrapped {
        jaxpr,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

fn build_ledger(jaxpr: Jaxpr, transforms: &[Transform]) -> TraceTransformLedger {
    let mut ledger = TraceTransformLedger::new(jaxpr);
    for (idx, transform) in transforms.iter().enumerate() {
        ledger.push_transform(*transform, format!("fj-api-{}-{}", transform.as_str(), idx));
    }
    ledger
}

fn dispatch_with(
    jaxpr: Jaxpr,
    transforms: &[Transform],
    args: Vec<Value>,
    backend: &str,
    mode: CompatibilityMode,
) -> Result<Vec<Value>, ApiError> {
    let response = dispatch(DispatchRequest {
        mode,
        ledger: build_ledger(jaxpr, transforms),
        args,
        backend: backend.to_owned(),
        compile_options: BTreeMap::new(),
        custom_hook: None,
        unknown_incompatible_features: vec![],
    })?;
    Ok(response.outputs)
}

/// Compose transforms: `jit(grad(f))` becomes `jit(jaxpr).compose_grad()`.
#[must_use]
pub fn compose(jaxpr: Jaxpr, transforms: Vec<Transform>) -> ComposedTransform {
    ComposedTransform {
        jaxpr,
        transforms,
        backend: "cpu".to_owned(),
        mode: CompatibilityMode::Strict,
    }
}

impl JitWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Jit],
            args,
            &self.backend,
            self.mode,
        )
    }

    /// Compose: `jit(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
        }
    }

    /// Compose: `jit(vmap(f))`.
    #[must_use]
    pub fn compose_vmap(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Jit, Transform::Vmap],
            backend: self.backend,
            mode: self.mode,
        }
    }
}

impl GradWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Grad],
            args,
            &self.backend,
            self.mode,
        )
    }
}

impl VmapWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set in_axes: comma-separated axis specs, e.g. "0,none,1".
    /// - Integer: batch along that axis
    /// - "none": this input is not batched (broadcast)
    #[must_use]
    pub fn with_in_axes(mut self, in_axes: &str) -> Self {
        self.in_axes = Some(in_axes.to_owned());
        self
    }

    /// Set out_axes: comma-separated axis specs for output batch dim placement.
    /// - Integer: place batch dim at that axis position
    /// - "none": output is not batched
    #[must_use]
    pub fn with_out_axes(mut self, out_axes: &str) -> Self {
        self.out_axes = Some(out_axes.to_owned());
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        let mut compile_options = BTreeMap::new();
        if let Some(ref in_axes) = self.in_axes {
            compile_options.insert("vmap_in_axes".to_owned(), in_axes.clone());
        }
        if let Some(ref out_axes) = self.out_axes {
            compile_options.insert("vmap_out_axes".to_owned(), out_axes.clone());
        }

        let response = dispatch(DispatchRequest {
            mode: self.mode,
            ledger: build_ledger(self.jaxpr.clone(), &[Transform::Vmap]),
            args,
            backend: self.backend.clone(),
            compile_options,
            custom_hook: None,
            unknown_incompatible_features: vec![],
        })?;
        Ok(response.outputs)
    }

    /// Compose: `vmap(grad(f))`.
    #[must_use]
    pub fn compose_grad(self) -> ComposedTransform {
        ComposedTransform {
            jaxpr: self.jaxpr,
            transforms: vec![Transform::Vmap, Transform::Grad],
            backend: self.backend,
            mode: self.mode,
        }
    }
}

impl ComposedTransform {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<Vec<Value>, ApiError> {
        dispatch_with(
            self.jaxpr.clone(),
            &self.transforms,
            args,
            &self.backend,
            self.mode,
        )
    }
}

impl ValueAndGradWrapped {
    #[must_use]
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_owned();
        self
    }

    #[must_use]
    pub fn with_mode(mut self, mode: CompatibilityMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn call(&self, args: Vec<Value>) -> Result<(Vec<Value>, Vec<Value>), ApiError> {
        let value = dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Jit],
            args.clone(),
            &self.backend,
            self.mode,
        )?;
        let gradient = dispatch_with(
            self.jaxpr.clone(),
            &[Transform::Grad],
            args,
            &self.backend,
            self.mode,
        )?;
        Ok((value, gradient))
    }
}
