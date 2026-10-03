//! Trusted in-process C ABI for the Python real-model evaluator.
//! Integer handles reject stale/double releases. Tensor pointers must describe
//! readable/writable arrays for the declared length for the duration of a call.
use crate::{Cache, Config, Error, Namespace, SequenceId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

struct Registry {
    next: u64,
    caches: HashMap<u64, Cache>,
}
fn registry() -> &'static Mutex<Registry> {
    static VALUE: OnceLock<Mutex<Registry>> = OnceLock::new();
    VALUE.get_or_init(|| {
        Mutex::new(Registry {
            next: 1,
            caches: HashMap::new(),
        })
    })
}

#[no_mangle]
pub extern "C" fn kw_new(
    pages: usize,
    page_tokens: usize,
    heads: usize,
    dim: usize,
    sequences: usize,
    entries: usize,
) -> u64 {
    let Ok(cache) = Cache::new(Config {
        pages,
        page_tokens,
        kv_heads: heads,
        head_dim: dim,
        max_sequences: sequences,
        cache_entries: entries,
    }) else {
        return 0;
    };
    let mut r = registry().lock().unwrap();
    let Some(next) = r.next.checked_add(1) else {
        return 0;
    };
    let id = r.next;
    r.next = next;
    r.caches.insert(id, cache);
    id
}
#[no_mangle]
pub extern "C" fn kw_drop(handle: u64) -> i32 {
    if registry().lock().unwrap().caches.remove(&handle).is_some() {
        0
    } else {
        -1
    }
}
/// # Safety
/// prompt must be readable for length u32 elements (unless length is zero), and reused writable for one usize.
#[no_mangle]
pub unsafe extern "C" fn kw_checkout(
    handle: u64,
    tenant: u64,
    prompt: *const u32,
    length: usize,
    reused: *mut usize,
) -> u64 {
    let mut r = registry().lock().unwrap();
    let Some(cache) = r.caches.get_mut(&handle) else {
        return 0;
    };
    if reused.is_null()
        || length > cache.config.pages * cache.config.page_tokens
        || (length > 0 && prompt.is_null())
    {
        return 0;
    }
    let tokens = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(prompt, length)
    };
    let namespace = Namespace {
        model: "stories260K-layer".into(),
        revision: "v1".into(),
        tenant: tenant.to_string(),
    };
    match cache.checkout(namespace, tokens) {
        Ok((id, n)) => {
            *reused = n;
            id.0
        }
        Err(_) => 0,
    }
}
/// # Safety
/// key and value must be readable for width f32 elements.
#[no_mangle]
pub unsafe extern "C" fn kw_append(
    handle: u64,
    seq: u64,
    token: u32,
    key: *const f32,
    value: *const f32,
    width: usize,
) -> i32 {
    let mut r = registry().lock().unwrap();
    let Some(c) = r.caches.get_mut(&handle) else {
        return -1;
    };
    if key.is_null() || value.is_null() || width != c.config.kv_heads * c.config.head_dim {
        return -2;
    }
    match c.append(
        SequenceId(seq),
        token,
        std::slice::from_raw_parts(key, width),
        std::slice::from_raw_parts(value, width),
    ) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}
/// # Safety
/// query must be readable and out writable for width f32 elements; the arrays must not alias.
#[no_mangle]
pub unsafe extern "C" fn kw_attention(
    handle: u64,
    seq: u64,
    query: *const f32,
    width: usize,
    out: *mut f32,
) -> i32 {
    let r = registry().lock().unwrap();
    let Some(c) = r.caches.get(&handle) else {
        return -1;
    };
    if query.is_null() || out.is_null() || width == 0 || width > 128 * c.config.head_dim {
        return -2;
    }
    match c.attention(SequenceId(seq), std::slice::from_raw_parts(query, width)) {
        Ok(values) => {
            std::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
            0
        }
        Err(_) => -3,
    }
}
#[no_mangle]
pub extern "C" fn kw_release(handle: u64, seq: u64) -> i32 {
    let mut r = registry().lock().unwrap();
    let Some(c) = r.caches.get_mut(&handle) else {
        return -1;
    };
    match c.release(SequenceId(seq)) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}
#[no_mangle]
pub extern "C" fn kw_fork(handle: u64, seq: u64) -> u64 {
    let mut r = registry().lock().unwrap();
    let Some(c) = r.caches.get_mut(&handle) else {
        return 0;
    };
    c.fork(SequenceId(seq)).map_or(0, |id| id.0)
}
#[no_mangle]
pub extern "C" fn kw_publish(handle: u64, seq: u64) -> i64 {
    let mut r = registry().lock().unwrap();
    let Some(c) = r.caches.get_mut(&handle) else {
        return -1;
    };
    match c.publish_prefix(SequenceId(seq)) {
        Ok(n) => n as i64,
        Err(Error::ConflictingPrefix) => -4,
        Err(_) => -3,
    }
}
#[no_mangle]
pub extern "C" fn kw_clear(handle: u64) -> i32 {
    let mut r = registry().lock().unwrap();
    let Some(c) = r.caches.get_mut(&handle) else {
        return -1;
    };
    c.clear_prefixes();
    0
}
/// # Safety
/// out must be writable for length u64 elements.
#[no_mangle]
pub unsafe extern "C" fn kw_stats(handle: u64, out: *mut u64, length: usize) -> i32 {
    let r = registry().lock().unwrap();
    let Some(c) = r.caches.get(&handle) else {
        return -1;
    };
    if out.is_null() || length != 8 {
        return -2;
    }
    if c.check_invariants().is_err() {
        return -3;
    }
    let s = c.stats();
    let fields = [
        s.allocated_payload_bytes as u64,
        s.allocated_pages as u64,
        s.sequences as u64,
        s.cached_prefixes as u64,
        s.prefix_hits,
        s.reused_tokens,
        s.cow_copies,
        s.evictions,
    ];
    std::ptr::copy_nonoverlapping(fields.as_ptr(), out, 8);
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stale_handles_and_null_buffers_fail_closed() {
        let h = kw_new(2, 4, 1, 2, 2, 1);
        assert_ne!(h, 0);
        assert_eq!(
            unsafe { kw_checkout(h, 0, std::ptr::null(), 1, std::ptr::null_mut()) },
            0
        );
        assert_eq!(
            unsafe { kw_append(h, 1, 0, std::ptr::null(), std::ptr::null(), 2) },
            -2
        );
        assert_eq!(kw_drop(h), 0);
        assert_eq!(kw_drop(h), -1);
        assert_eq!(kw_release(h, 1), -1);
    }
}
