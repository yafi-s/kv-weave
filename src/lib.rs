//! Bounded, single-layer KV storage and numerically stable CPU attention.
//! Ownership is explicit: sequences and cached prefixes each hold page references.
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Namespace {
    pub model: String,
    pub revision: String,
    pub tenant: String,
}
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub pages: usize,
    pub page_tokens: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub max_sequences: usize,
    pub cache_entries: usize,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidConfig,
    InvalidNamespace,
    InvalidShape,
    NonFinite,
    UnknownSequence,
    OutOfPages,
    SequenceLimit,
    EmptySequence,
    CacheDisabled,
    EmptyPrefix,
    ConflictingPrefix,
    IdExhausted,
    Invariant,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);
#[derive(Clone)]
struct Sequence {
    pages: Vec<usize>,
    tokens: Vec<u32>,
    namespace: Namespace,
}
struct Page {
    refs: usize,
    data: Vec<f32>,
}
#[derive(Clone, PartialEq, Eq, Hash)]
struct Key {
    namespace: Namespace,
    tokens: Vec<u32>,
}
struct Prefix {
    pages: Vec<usize>,
    touched: u128,
}
#[derive(Debug, Clone, Copy)]
pub struct Stats {
    pub allocated_pages: usize,
    pub logical_tokens: usize,
    pub sequences: usize,
    pub cached_prefixes: usize,
    pub prefix_hits: u64,
    pub reused_tokens: u64,
    pub cow_copies: u64,
    pub evictions: u64,
    pub allocated_payload_bytes: usize,
}
pub struct Cache {
    config: Config,
    pages: Vec<Page>,
    free: Vec<usize>,
    sequences: HashMap<SequenceId, Sequence>,
    prefixes: HashMap<Key, Prefix>,
    next_id: u64,
    clock: u128,
    hits: u64,
    reused: u64,
    copies: u64,
    evictions: u64,
}
impl Cache {
    pub fn new(config: Config) -> Result<Self> {
        let c = config;
        if c.pages == 0
            || c.pages > 4096
            || c.page_tokens == 0
            || c.page_tokens > 1024
            || c.kv_heads == 0
            || c.kv_heads > 128
            || c.head_dim == 0
            || c.head_dim > 512
            || c.max_sequences == 0
            || c.max_sequences > 128
            || c.cache_entries > 128
        {
            return Err(Error::InvalidConfig);
        }
        let slots = c.pages * c.page_tokens;
        let payload = slots
            .checked_mul(c.kv_heads)
            .and_then(|n| n.checked_mul(c.head_dim))
            .and_then(|n| n.checked_mul(2))
            .ok_or(Error::InvalidConfig)?;
        if payload > 64 * 1024 * 1024
            || slots * (c.max_sequences + c.cache_entries) > 16 * 1024 * 1024
        {
            return Err(Error::InvalidConfig);
        }
        Ok(Self {
            config,
            pages: (0..c.pages)
                .map(|_| Page {
                    refs: 0,
                    data: Vec::new(),
                })
                .collect(),
            free: (0..c.pages).rev().collect(),
            sequences: HashMap::new(),
            prefixes: HashMap::new(),
            next_id: 1,
            clock: 0,
            hits: 0,
            reused: 0,
            copies: 0,
            evictions: 0,
        })
    }
    fn tick(&mut self) -> u128 {
        self.clock += 1;
        self.clock
    }
    fn new_id(&mut self) -> Result<SequenceId> {
        if self.sequences.len() >= self.config.max_sequences {
            return Err(Error::SequenceLimit);
        }
        let next = self.next_id.checked_add(1).ok_or(Error::IdExhausted)?;
        let id = SequenceId(self.next_id);
        self.next_id = next;
        Ok(id)
    }
    fn drop_page(&mut self, id: usize) {
        self.pages[id].refs -= 1;
        if self.pages[id].refs == 0 {
            self.pages[id].data = Vec::new();
            self.free.push(id)
        }
    }
    fn evict_one(&mut self) -> bool {
        let key = self
            .prefixes
            .iter()
            .min_by_key(|(_, v)| v.touched)
            .map(|(k, _)| k.clone());
        if let Some(key) = key {
            let old = self.prefixes.remove(&key).unwrap();
            for id in old.pages {
                self.drop_page(id)
            }
            self.evictions += 1;
            true
        } else {
            false
        }
    }
    fn allocate(&mut self) -> Result<usize> {
        while self.free.is_empty() {
            if !self.evict_one() {
                return Err(Error::OutOfPages);
            }
        }
        let id = self.free.pop().unwrap();
        let c = self.config;
        self.pages[id].data = vec![0.; c.page_tokens * c.kv_heads * c.head_dim * 2];
        self.pages[id].refs = 1;
        Ok(id)
    }
    /// Reuse the longest cached full-page prefix. Only reused tokens are installed;
    /// the caller must compute and append the uncached suffix of the prompt.
    pub fn checkout(
        &mut self,
        namespace: Namespace,
        prompt: &[u32],
    ) -> Result<(SequenceId, usize)> {
        if [&namespace.model, &namespace.revision, &namespace.tenant]
            .iter()
            .any(|s| s.is_empty() || s.len() > 256)
        {
            return Err(Error::InvalidNamespace);
        }
        if prompt.len() > self.config.pages * self.config.page_tokens {
            return Err(Error::InvalidShape);
        }
        let id = self.new_id()?;
        // Bounded entry scan avoids allocating every possible prompt prefix on
        // misses. A radix index would be a separate scalability optimization.
        let matched = self
            .prefixes
            .keys()
            .filter(|key| key.namespace == namespace && prompt.starts_with(&key.tokens))
            .max_by_key(|key| key.tokens.len())
            .cloned();
        let (length, owned) = if let Some(key) = matched {
            let tick = self.tick();
            let entry = self.prefixes.get_mut(&key).unwrap();
            entry.touched = tick;
            (key.tokens.len(), entry.pages.clone())
        } else {
            (0, Vec::new())
        };
        for &page in &owned {
            self.pages[page].refs += 1
        }
        if length > 0 {
            self.hits += 1;
            self.reused += length as u64
        }
        self.sequences.insert(
            id,
            Sequence {
                pages: owned,
                tokens: prompt[..length].to_vec(),
                namespace,
            },
        );
        Ok((id, length))
    }
    pub fn fork(&mut self, parent: SequenceId) -> Result<SequenceId> {
        let sequence = self
            .sequences
            .get(&parent)
            .ok_or(Error::UnknownSequence)?
            .clone();
        let id = self.new_id()?;
        for &p in &sequence.pages {
            self.pages[p].refs += 1
        }
        self.sequences.insert(id, sequence);
        Ok(id)
    }
    /// On page exhaustion, active sequence contents stay unchanged. Cold prefix
    /// entries may have been evicted while attempting admission. Allocator OOM
    /// is still process-fatal, as with ordinary Rust Vec allocations.
    pub fn append(&mut self, id: SequenceId, token: u32, key: &[f32], value: &[f32]) -> Result<()> {
        let c = self.config;
        let width = c.kv_heads * c.head_dim;
        if key.len() != width || value.len() != width {
            return Err(Error::InvalidShape);
        }
        if key.iter().chain(value).any(|x| !x.is_finite()) {
            return Err(Error::NonFinite);
        }
        let seq = self.sequences.get(&id).ok_or(Error::UnknownSequence)?;
        let offset = seq.tokens.len() % c.page_tokens;
        let old = seq.pages.last().copied();
        let page = if offset == 0 {
            self.allocate()?
        } else if self.pages[old.unwrap()].refs > 1 {
            let new = self.allocate()?;
            let old = old.unwrap();
            // Copy only the initialized prefix of a partially filled shared page.
            let used = offset * width * 2;
            let (left, right) = if old < new {
                let (a, b) = self.pages.split_at_mut(new);
                (&a[old].data, &mut b[0].data)
            } else {
                let (a, b) = self.pages.split_at_mut(old);
                (&b[0].data, &mut a[new].data)
            };
            right[..used].copy_from_slice(&left[..used]);
            self.drop_page(old);
            self.copies += 1;
            new
        } else {
            old.unwrap()
        };
        let seq = self.sequences.get_mut(&id).unwrap();
        if offset == 0 {
            seq.pages.push(page)
        } else {
            *seq.pages.last_mut().unwrap() = page
        }
        let start = offset * width * 2;
        self.pages[page].data[start..start + width].copy_from_slice(key);
        self.pages[page].data[start + width..start + width * 2].copy_from_slice(value);
        seq.tokens.push(token);
        Ok(())
    }
    /// Cache the largest complete-page prefix. Namespace and exact token vectors
    /// are part of the key. Publishing conflicting KV values for the same key is
    /// rejected instead of silently poisoning future readers.
    pub fn publish_prefix(&mut self, id: SequenceId) -> Result<usize> {
        if self.config.cache_entries == 0 {
            return Err(Error::CacheDisabled);
        }
        let seq = self.sequences.get(&id).ok_or(Error::UnknownSequence)?;
        let blocks = seq.tokens.len() / self.config.page_tokens;
        if blocks == 0 {
            return Err(Error::EmptyPrefix);
        }
        let length = blocks * self.config.page_tokens;
        let key = Key {
            namespace: seq.namespace.clone(),
            tokens: seq.tokens[..length].to_vec(),
        };
        let pages = seq.pages[..blocks].to_vec();
        if let Some(existing) = self.prefixes.get(&key) {
            if existing
                .pages
                .iter()
                .zip(&pages)
                .any(|(&a, &b)| self.pages[a].data != self.pages[b].data)
            {
                return Err(Error::ConflictingPrefix);
            }
            let tick = self.tick();
            self.prefixes.get_mut(&key).unwrap().touched = tick;
            return Ok(length);
        }
        // Pin first: eviction is allowed to remove other references to these pages.
        for &p in &pages {
            self.pages[p].refs += 1
        }
        if self.prefixes.len() >= self.config.cache_entries {
            self.evict_one();
        }
        let touched = self.tick();
        self.prefixes.insert(key, Prefix { pages, touched });
        Ok(length)
    }
    pub fn release(&mut self, id: SequenceId) -> Result<()> {
        let seq = self.sequences.remove(&id).ok_or(Error::UnknownSequence)?;
        for p in seq.pages {
            self.drop_page(p)
        }
        Ok(())
    }
    pub fn clear_prefixes(&mut self) {
        while self.evict_one() {}
    }
    pub fn tokens(&self, id: SequenceId) -> Result<&[u32]> {
        Ok(&self
            .sequences
            .get(&id)
            .ok_or(Error::UnknownSequence)?
            .tokens)
    }
    /// Single-position causal attention over the sequence's stored prefix. GQA
    /// maps contiguous groups of query heads to one KV head. Accumulation uses
    /// f64 online softmax; storage and returned vectors use f32.
    pub fn attention(&self, id: SequenceId, query: &[f32]) -> Result<Vec<f32>> {
        let c = self.config;
        let seq = self.sequences.get(&id).ok_or(Error::UnknownSequence)?;
        if seq.tokens.is_empty() {
            return Err(Error::EmptySequence);
        }
        if query.is_empty() || query.len() % c.head_dim != 0 {
            return Err(Error::InvalidShape);
        }
        let qh = query.len() / c.head_dim;
        if qh > 128 || qh % c.kv_heads != 0 {
            return Err(Error::InvalidShape);
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFinite);
        }
        let mut result = vec![0.; query.len()];
        let width = c.kv_heads * c.head_dim;
        let scale = 1. / (c.head_dim as f64).sqrt();
        for head in 0..qh {
            let kv = head / (qh / c.kv_heads);
            let mut maximum = f64::NEG_INFINITY;
            let mut denominator = 0.;
            let mut acc = vec![0.; c.head_dim];
            for position in 0..seq.tokens.len() {
                let data = &self.pages[seq.pages[position / c.page_tokens]].data;
                let base = position % c.page_tokens * width * 2 + kv * c.head_dim;
                let score = (0..c.head_dim)
                    .map(|j| f64::from(query[head * c.head_dim + j]) * f64::from(data[base + j]))
                    .sum::<f64>()
                    * scale;
                let new_max = maximum.max(score);
                let old_scale = (maximum - new_max).exp();
                let weight = (score - new_max).exp();
                denominator = denominator * old_scale + weight;
                for j in 0..c.head_dim {
                    acc[j] = acc[j] * old_scale + weight * f64::from(data[base + width + j])
                }
                maximum = new_max;
            }
            for j in 0..c.head_dim {
                result[head * c.head_dim + j] = (acc[j] / denominator) as f32
            }
        }
        Ok(result)
    }
    pub fn stats(&self) -> Stats {
        let allocated = self.pages.iter().filter(|p| p.refs > 0).count();
        let c = self.config;
        Stats {
            allocated_pages: allocated,
            logical_tokens: self.sequences.values().map(|s| s.tokens.len()).sum(),
            sequences: self.sequences.len(),
            cached_prefixes: self.prefixes.len(),
            prefix_hits: self.hits,
            reused_tokens: self.reused,
            cow_copies: self.copies,
            evictions: self.evictions,
            allocated_payload_bytes: allocated
                * c.page_tokens
                * c.kv_heads
                * c.head_dim
                * 2
                * std::mem::size_of::<f32>(),
        }
    }
    pub fn check_invariants(&self) -> Result<()> {
        let mut expected = vec![0; self.pages.len()];
        let c = self.config;
        for seq in self.sequences.values() {
            if seq.pages.len() != seq.tokens.len().div_ceil(c.page_tokens) {
                return Err(Error::Invariant);
            }
            let unique: HashSet<_> = seq.pages.iter().collect();
            if unique.len() != seq.pages.len() {
                return Err(Error::Invariant);
            }
            for &p in &seq.pages {
                expected[p] += 1
            }
        }
        for (key, prefix) in &self.prefixes {
            if key.tokens.len() != prefix.pages.len() * c.page_tokens {
                return Err(Error::Invariant);
            }
            for &p in &prefix.pages {
                expected[p] += 1
            }
        }
        let free: HashSet<_> = self.free.iter().copied().collect();
        if free.len() != self.free.len() {
            return Err(Error::Invariant);
        }
        for (i, page) in self.pages.iter().enumerate() {
            if page.refs != expected[i] || free.contains(&i) != (page.refs == 0) {
                return Err(Error::Invariant);
            }
            if page.refs > 0 && page.data.len() != c.page_tokens * c.kv_heads * c.head_dim * 2 {
                return Err(Error::Invariant);
            }
        }
        Ok(())
    }
}
