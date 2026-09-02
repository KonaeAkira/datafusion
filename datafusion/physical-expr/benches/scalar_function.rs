// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Benchmarks for [`ScalarFunctionExpr::evaluate`].
//!
//! These measure the per-batch overhead that `evaluate` pays *before* the
//! function itself runs, which depends on the shape of the argument
//! expressions rather than on the function:
//!
//! * a plain `Column` argument -- `Column` overrides `return_field`, so this
//!   is the cheap baseline,
//! * a deep arithmetic tree -- `BinaryExpr` uses the default `return_field`,
//!   which renders the whole subtree with `Display` and walks it twice,
//! * `CASE WHEN col = lit THEN col END` -- `CaseExpr` also uses the default
//!   `return_field`, and its `nullable()` additionally tries to const-fold
//!   every `WHEN` predicate.

use std::hint::black_box;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion_common::config::ConfigOptions;
use datafusion_expr::Operator;
use datafusion_physical_expr::ScalarFunctionExpr;
use datafusion_physical_expr::expressions::{BinaryExpr, case, col, lit};
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

const BATCH_SIZE: usize = 8192;

/// A batch with a nullable `f64` column `c1` and a nullable utf8 column `s`.
fn make_batch() -> RecordBatch {
    let c1: Float64Array = (0..BATCH_SIZE).map(|i| Some(i as f64)).collect();
    let s: StringArray = (0..BATCH_SIZE)
        .map(|i| Some(if i % 3 == 0 { "lit" } else { "other" }))
        .collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("c1", c1.data_type().clone(), true),
        Field::new("s", s.data_type().clone(), true),
    ]));

    RecordBatch::try_new(schema, vec![Arc::new(c1), Arc::new(s)]).unwrap()
}

/// `abs(arg)`
fn abs(arg: Arc<dyn PhysicalExpr>, schema: &Schema) -> Arc<dyn PhysicalExpr> {
    Arc::new(
        ScalarFunctionExpr::try_new(
            datafusion_functions::math::abs(),
            vec![arg],
            schema,
            Arc::new(ConfigOptions::default()),
        )
        .unwrap(),
    )
}

/// `((((c1 + 1) * 2) - 3) / 4) ...`, `depth` operators deep.
fn deep_tree(c1: &Arc<dyn PhysicalExpr>, depth: usize) -> Arc<dyn PhysicalExpr> {
    let ops = [
        Operator::Plus,
        Operator::Multiply,
        Operator::Minus,
        Operator::Divide,
    ];
    let mut expr = Arc::clone(c1);
    for i in 0..depth {
        expr = Arc::new(BinaryExpr::new(expr, ops[i % ops.len()], lit(2.0f64)));
    }
    expr
}

fn criterion_benchmark(c: &mut Criterion) {
    let batch = make_batch();
    let schema = batch.schema();
    let c1 = col("c1", &schema).unwrap();
    let s = col("s", &schema).unwrap();

    let mut group = c.benchmark_group("scalar_function");

    // Cheap baseline: `Column` has a `return_field` override.
    group.bench_function("abs(c1)", |b| {
        let expr = abs(Arc::clone(&c1), &schema);
        b.iter(|| black_box(expr.evaluate(black_box(&batch)).unwrap()))
    });

    // Default `return_field`: `Display` render + two recursive walks per batch.
    for depth in [2, 8] {
        group.bench_function(format!("abs(<{depth} deep arithmetic tree>)"), |b| {
            let expr = abs(deep_tree(&c1, depth), &schema);
            b.iter(|| black_box(expr.evaluate(black_box(&batch)).unwrap()))
        });
    }

    // `CaseExpr::nullable` additionally const-folds the `WHEN` predicate, which
    // cannot succeed against the empty schema it uses and so builds and drops a
    // `DataFusionError` per batch.
    group.bench_function("abs(CASE WHEN s = 'lit' THEN c1 END)", |b| {
        let when = Arc::new(BinaryExpr::new(Arc::clone(&s), Operator::Eq, lit("lit")));
        let case_expr = case(None, vec![(when, Arc::clone(&c1))], None).unwrap();
        let expr = abs(case_expr, &schema);
        b.iter(|| black_box(expr.evaluate(black_box(&batch)).unwrap()))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
