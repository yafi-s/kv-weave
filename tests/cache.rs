use kv_weave::{Cache, Config, Error, Namespace};
fn namespace(tenant: &str) -> Namespace {
    Namespace {
        model: "tiny-reference".into(),
        revision: "weights-v1".into(),
        tenant: tenant.into(),
    }
}
fn config() -> Config {
    Config {
        pages: 32,
        page_tokens: 4,
        kv_heads: 2,
        head_dim: 4,
        max_sequences: 32,
        cache_entries: 8,
    }
}
fn tensor(seed: u32, width: usize) -> Vec<f32> {
    (0..width)
        .map(|i| (((seed as usize * 17 + i * 31) % 101) as f32 - 50.) / 20.)
        .collect()
}
fn append(cache: &mut Cache, id: kv_weave::SequenceId, token: u32) {
    cache
        .append(id, token, &tensor(token, 8), &tensor(token + 7, 8))
        .unwrap()
}

// Independent dense oracle: materialize every logit, find its maximum in a
// separate pass, normalize once, and then take a weighted sum of all values.
fn dense(tokens: &[u32], query: &[f32], kv_heads: usize, dim: usize) -> Vec<f32> {
    let qh = query.len() / dim;
    let mut out = vec![0.; query.len()];
    for h in 0..qh {
        let kv = h / (qh / kv_heads);
        let logits: Vec<_> = tokens
            .iter()
            .map(|&t| {
                let key = tensor(t, kv_heads * dim);
                (0..dim)
                    .map(|j| f64::from(query[h * dim + j]) * f64::from(key[kv * dim + j]))
                    .sum::<f64>()
                    / (dim as f64).sqrt()
            })
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<_> = logits.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for j in 0..dim {
            out[h * dim + j] = (tokens
                .iter()
                .zip(&weights)
                .map(|(&t, &w)| w * f64::from(tensor(t + 7, kv_heads * dim)[kv * dim + j]))
                .sum::<f64>()
                / sum) as f32
        }
    }
    out
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (&x, &y) in a.iter().zip(b) {
        assert!((x - y).abs() < 2e-5, "{x} vs {y}")
    }
}

#[test]
fn grouped_attention_matches_dense_across_page_boundaries() {
    for len in [1, 3, 4, 5, 17, 63] {
        let mut c = Cache::new(config()).unwrap();
        let (id, _) = c.checkout(namespace("a"), &[]).unwrap();
        let tokens: Vec<_> = (0..len).collect();
        for &t in &tokens {
            append(&mut c, id, t)
        }
        for heads in [2, 4, 8] {
            let q = tensor(11, heads * 4);
            close(&c.attention(id, &q).unwrap(), &dense(&tokens, &q, 2, 4));
        }
        c.check_invariants().unwrap();
    }
}
#[test]
fn fork_partial_page_is_copy_on_write() {
    let mut c = Cache::new(config()).unwrap();
    let (parent, _) = c.checkout(namespace("a"), &[]).unwrap();
    for t in 0..3 {
        append(&mut c, parent, t)
    }
    let child = c.fork(parent).unwrap();
    append(&mut c, parent, 90);
    append(&mut c, child, 91);
    assert_eq!(c.tokens(parent).unwrap(), [0, 1, 2, 90]);
    assert_eq!(c.tokens(child).unwrap(), [0, 1, 2, 91]);
    let q = tensor(4, 16);
    close(
        &c.attention(parent, &q).unwrap(),
        &dense(&[0, 1, 2, 90], &q, 2, 4),
    );
    close(
        &c.attention(child, &q).unwrap(),
        &dense(&[0, 1, 2, 91], &q, 2, 4),
    );
    assert_eq!(c.stats().cow_copies, 1);
    c.release(parent).unwrap();
    c.release(child).unwrap();
    c.check_invariants().unwrap();
    assert_eq!(c.stats().allocated_pages, 0);
    assert_eq!(c.release(child), Err(Error::UnknownSequence));
}
#[test]
fn longest_prefix_and_namespace_isolation() {
    let mut c = Cache::new(config()).unwrap();
    let (id, _) = c.checkout(namespace("a"), &[]).unwrap();
    for t in 0..4 {
        append(&mut c, id, t)
    }
    c.publish_prefix(id).unwrap();
    for t in 4..8 {
        append(&mut c, id, t)
    }
    c.publish_prefix(id).unwrap();
    c.release(id).unwrap();
    let prompt: Vec<_> = (0..10).collect();
    let (hit, reused) = c.checkout(namespace("a"), &prompt).unwrap();
    assert_eq!(reused, 8);
    let (_, short) = c.checkout(namespace("a"), &[0, 1, 2, 3, 99]).unwrap();
    assert_eq!(short, 4);
    let (_, miss) = c.checkout(namespace("b"), &prompt).unwrap();
    assert_eq!(miss, 0);
    let mut revision = namespace("a");
    revision.revision = "weights-v2".into();
    assert_eq!(c.checkout(revision, &prompt).unwrap().1, 0);
    for t in 8..10 {
        append(&mut c, hit, t)
    }
    let q = tensor(2, 16);
    close(&c.attention(hit, &q).unwrap(), &dense(&prompt, &q, 2, 4));
    c.check_invariants().unwrap();
}
#[test]
fn memory_pressure_evicts_only_cache_ownership() {
    let mut cfg = config();
    cfg.pages = 2;
    let mut c = Cache::new(cfg).unwrap();
    let (a, _) = c.checkout(namespace("a"), &[]).unwrap();
    for t in 0..4 {
        append(&mut c, a, t)
    }
    c.publish_prefix(a).unwrap();
    c.release(a).unwrap();
    let (b, _) = c.checkout(namespace("b"), &[]).unwrap();
    for t in 20..28 {
        append(&mut c, b, t)
    }
    assert_eq!(c.stats().evictions, 1);
    assert_eq!(c.stats().allocated_pages, 2);
    let saved = c.tokens(b).unwrap().to_vec();
    assert_eq!(
        c.append(b, 99, &tensor(99, 8), &tensor(106, 8)),
        Err(Error::OutOfPages)
    );
    assert_eq!(c.tokens(b).unwrap(), saved);
    c.check_invariants().unwrap();
}
#[test]
fn cow_exhaustion_does_not_mutate_live_sequences() {
    let mut cfg = config();
    cfg.pages = 1;
    let mut c = Cache::new(cfg).unwrap();
    let (a, _) = c.checkout(namespace("a"), &[]).unwrap();
    append(&mut c, a, 1);
    let b = c.fork(a).unwrap();
    assert_eq!(
        c.append(b, 2, &tensor(2, 8), &tensor(9, 8)),
        Err(Error::OutOfPages)
    );
    assert_eq!(c.tokens(a).unwrap(), [1]);
    assert_eq!(c.tokens(b).unwrap(), [1]);
    c.release(a).unwrap();
    append(&mut c, b, 2);
    assert_eq!(c.stats().cow_copies, 0);
    c.check_invariants().unwrap();
}
#[test]
fn conflicting_prefix_rejected_and_nan_never_enters_cache() {
    let mut c = Cache::new(config()).unwrap();
    let (a, _) = c.checkout(namespace("a"), &[]).unwrap();
    let (b, _) = c.checkout(namespace("a"), &[]).unwrap();
    for t in 0..4 {
        append(&mut c, a, t);
        c.append(b, t, &[10.; 8], &[20.; 8]).unwrap()
    }
    c.publish_prefix(a).unwrap();
    assert_eq!(c.publish_prefix(b), Err(Error::ConflictingPrefix));
    assert_eq!(
        c.append(a, 4, &[f32::NAN; 8], &[0.; 8]),
        Err(Error::NonFinite)
    );
    assert_eq!(c.attention(a, &[f32::INFINITY; 8]), Err(Error::NonFinite));
    assert_eq!(c.attention(a, &[0.; 3]), Err(Error::InvalidShape));
    c.check_invariants().unwrap();
}
#[test]
fn extreme_finite_logits_remain_finite() {
    let mut c = Cache::new(config()).unwrap();
    let (a, _) = c.checkout(namespace("a"), &[]).unwrap();
    c.append(a, 1, &[1e30; 8], &[3.; 8]).unwrap();
    c.append(a, 2, &[-1e30; 8], &[9.; 8]).unwrap();
    let out = c.attention(a, &[1e30; 16]).unwrap();
    assert!(out.iter().all(|x| x.is_finite() && (*x - 3.).abs() < 1e-6));
}
#[test]
fn deterministic_random_ownership_state_machine() {
    let mut c = Cache::new(config()).unwrap();
    let mut active = Vec::new();
    let mut histories = std::collections::HashMap::new();
    let mut rng = 17u64;
    let random = |r: &mut u64| {
        *r ^= *r << 13;
        *r ^= *r >> 7;
        *r ^= *r << 17;
        *r
    };
    for _ in 0..10000 {
        let choice = random(&mut rng) % 5;
        if active.is_empty() || (choice == 0 && active.len() < 20) {
            let (id, _) = c.checkout(namespace("state-machine"), &[]).unwrap();
            active.push(id);
            histories.insert(id, Vec::new());
        } else {
            let pos = random(&mut rng) as usize % active.len();
            let id = active[pos];
            match choice {
                1 if active.len() < 20 => {
                    let child = c.fork(id).unwrap();
                    histories.insert(child, histories[&id].clone());
                    active.push(child);
                }
                2 => {
                    c.release(id).unwrap();
                    active.swap_remove(pos);
                    histories.remove(&id);
                }
                3 => {
                    let _ = c.publish_prefix(id);
                }
                _ => {
                    let token = (random(&mut rng) % 1000) as u32;
                    match c.append(id, token, &tensor(token, 8), &tensor(token + 7, 8)) {
                        Ok(()) => histories.get_mut(&id).unwrap().push(token),
                        Err(Error::OutOfPages) => {}
                        other => panic!("{other:?}"),
                    }
                }
            }
        }
        c.check_invariants().unwrap();
        for &id in &active {
            assert_eq!(c.tokens(id).unwrap(), histories[&id]);
            if !histories[&id].is_empty() {
                let q = tensor(12, 8);
                close(
                    &c.attention(id, &q).unwrap(),
                    &dense(&histories[&id], &q, 2, 4),
                );
            }
        }
    }
    for id in active {
        c.release(id).unwrap()
    }
    c.clear_prefixes();
    assert_eq!(c.stats().allocated_pages, 0);
    c.check_invariants().unwrap();
}
