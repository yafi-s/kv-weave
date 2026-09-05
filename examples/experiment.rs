use kv_weave::{Cache, Config, Namespace};
use std::time::Instant;
fn tensor(token: u32) -> Vec<f32> {
    (0..64)
        .map(|i| (((token as usize * 13 + i * 7) % 97) as f32 - 48.) / 32.)
        .collect()
}
fn main() {
    let namespace = Namespace {
        model: "reference-projection".into(),
        revision: "v1".into(),
        tenant: "experiment".into(),
    };
    let prompt: Vec<u32> = (0..64).collect();
    let query = tensor(700);
    let mut baseline = None;
    for shared in [false, true] {
        let cfg = Config {
            pages: 192,
            page_tokens: 16,
            kv_heads: 2,
            head_dim: 32,
            max_sequences: 64,
            cache_entries: 8,
        };
        let mut cache = Cache::new(cfg).unwrap();
        let start = Instant::now();
        let mut appended = 0;
        if shared {
            let (seed, _) = cache.checkout(namespace.clone(), &[]).unwrap();
            for &t in &prompt {
                cache.append(seed, t, &tensor(t), &tensor(t + 3)).unwrap();
                appended += 1;
            }
            cache.publish_prefix(seed).unwrap();
            cache.release(seed).unwrap();
        }
        let mut sequences = Vec::new();
        for request in 0..32 {
            let (id, reused) = cache
                .checkout(namespace.clone(), if shared { &prompt } else { &[] })
                .unwrap();
            for &t in &prompt[reused..] {
                cache.append(id, t, &tensor(t), &tensor(t + 3)).unwrap();
                appended += 1;
            }
            for t in 64 + request * 8..72 + request * 8 {
                cache.append(id, t, &tensor(t), &tensor(t + 3)).unwrap();
                appended += 1;
            }
            sequences.push(id);
        }
        let prepare_us = start.elapsed().as_micros();
        let stats = cache.stats();
        let start = Instant::now();
        let mut outputs = Vec::new();
        for &id in &sequences {
            outputs.push(cache.attention(id, &query).unwrap())
        }
        let attention_us = start.elapsed().as_micros();
        if let Some(previous) = &baseline {
            assert_eq!(previous, &outputs)
        } else {
            baseline = Some(outputs.clone())
        }
        let checksum: f64 = outputs.iter().flatten().map(|&v| f64::from(v)).sum();
        cache.check_invariants().unwrap();
        println!("{{\"mode\":\"{}\",\"requests\":32,\"tokens_per_request\":72,\"appended_tokens\":{appended},\"allocated_pages\":{},\"payload_bytes\":{},\"reused_tokens\":{},\"prepare_us\":{prepare_us},\"attention_us\":{attention_us},\"checksum\":{checksum}}}",if shared{"prefix-sharing"}else{"cold"},stats.allocated_pages,stats.allocated_payload_bytes,stats.reused_tokens);
        for id in sequences {
            cache.release(id).unwrap()
        }
        cache.clear_prefixes();
        cache.check_invariants().unwrap();
        assert_eq!(cache.stats().allocated_pages, 0);
    }
}
