fn main() {
    println!("Hello, world!");
}

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// head and tail on separate cache lines: no false sharing between cores.
#[repr(align(64))]
struct CachePadded<T>(T);

struct Ring<T, const N: usize> {
    head: CachePadded<AtomicUsize>, // next slot to read; only the consumer writes it
    tail: CachePadded<AtomicUsize>, // next slot to write; only the producer writes it
    slots: [UnsafeCell<MaybeUninit<T>>; N],
}

// SAFETY: slots are only touched through Producer (indices [tail, head+N)) and
// Consumer (indices [head, tail)). Those ranges never overlap, and ownership of
// a slot moves between them only via the Release/Acquire pairs on head and tail,
// so no slot is ever accessed by both threads without a happens-before edge.
// T: Send because values written on one thread are read (and dropped) on another.
unsafe impl<T: Send, const N: usize> Sync for Ring<T, N> {}

impl<T, const N: usize> Ring<T, N> {
    const MASK: usize = N - 1; // N is a power of two, so i & MASK == i % N
}

impl<T, const N: usize> Drop for Ring<T, N> {
    fn drop(&mut self) {
        let (mut h, t) = (*self.head.0.get_mut(), *self.tail.0.get_mut());
        while h != t {
            // SAFETY: &mut self means no Producer/Consumer is alive, and slots in
            // [head, tail) were initialized by push and not yet moved out by pop.
            unsafe { self.slots[h & Self::MASK].get_mut().assume_init_drop() };
            h = h.wrapping_add(1);
        }
    }
}

/// Not Clone: the type system enforces one producer and one consumer.
pub struct Producer<T, const N: usize>(Arc<Ring<T, N>>);
pub struct Consumer<T, const N: usize>(Arc<Ring<T, N>>);

pub fn channel<T, const N: usize>() -> (Producer<T, N>, Consumer<T, N>) {
    assert!(N.is_power_of_two(), "capacity must be a power of two");
    let ring = Arc::new(Ring {
        head: CachePadded(AtomicUsize::new(0)),
        tail: CachePadded(AtomicUsize::new(0)),
        slots: std::array::from_fn(|_| UnsafeCell::new(MaybeUninit::uninit())),
    });
    (Producer(ring.clone()), Consumer(ring))
}

impl<T, const N: usize> Producer<T, N> {
    /// Never blocks or allocates; hands the value back when full.
    pub fn push(&mut self, v: T) -> Result<(), T> {
        let r = &*self.0;
        let tail = r.tail.0.load(Ordering::Relaxed); // we are the only writer
        let head = r.head.0.load(Ordering::Acquire); // see slots the consumer freed
        if tail.wrapping_sub(head) == N {
            return Err(v);
        }
        // SAFETY: tail - head < N, so slot `tail` is outside [head, tail) and the
        // consumer will not touch it until we publish the new tail below. The
        // Acquire on head ordered this write after the consumer's last read of
        // the slot. Producer is not Clone and push takes &mut self, so we are the
        // only writer. The slot holds no live value (never written, or already
        // moved out by pop), so overwriting without dropping leaks nothing.
        unsafe { (*r.slots[tail & Ring::<T, N>::MASK].get()).write(v) };
        r.tail.0.store(tail.wrapping_add(1), Ordering::Release); // publish
        Ok(())
    }
}

impl<T, const N: usize> Consumer<T, N> {
    pub fn pop(&mut self) -> Option<T> {
        let r = &*self.0;
        let head = r.head.0.load(Ordering::Relaxed); // we are the only writer
        let tail = r.tail.0.load(Ordering::Acquire); // see the producer's write
        if head == tail {
            return None;
        }
        // SAFETY: head != tail, so slot `head` is in [head, tail): the producer
        // initialized it and will not touch it until we publish the new head
        // below. The Acquire on tail makes that write visible. Consumer is not
        // Clone and pop takes &mut self, so this value is read out exactly once.
        let v = unsafe { (*r.slots[head & Ring::<T, N>::MASK].get()).assume_init_read() };
        r.head.0.store(head.wrapping_add(1), Ordering::Release); // free the slot
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::thread;

    // Miri is ~1000x slower; shrink the workloads there.
    const ITEMS: usize = if cfg!(miri) { 500 } else { 200_000 };
    const MODEL_OPS: usize = if cfg!(miri) { 300 } else { 20_000 };
    const SEEDS: u64 = if cfg!(miri) { 2 } else { 16 };

    /// Same as `channel`, but with head and tail starting at `start`, so tests can
    /// cross the usize::MAX -> 0 wrap without pushing 2^64 items.
    fn channel_at<T, const N: usize>(start: usize) -> (Producer<T, N>, Consumer<T, N>) {
        assert!(N.is_power_of_two());
        let ring = Arc::new(Ring {
            head: CachePadded(AtomicUsize::new(start)),
            tail: CachePadded(AtomicUsize::new(start)),
            slots: std::array::from_fn(|_| UnsafeCell::new(MaybeUninit::uninit())),
        });
        (Producer(ring.clone()), Consumer(ring))
    }

    /// Counts its own drops, so tests can prove every value is dropped exactly once.
    #[derive(Debug)]
    struct Counted {
        id: usize,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for Counted {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    fn push_spin<T, const N: usize>(p: &mut Producer<T, N>, mut v: T) {
        while let Err(back) = p.push(v) {
            v = back;
            thread::yield_now();
        }
    }

    /// Spins until a value arrives, but fails instead of hanging if the producer
    /// thread exits (or panics) first.
    fn pop_or_fail<T, const N: usize>(
        c: &mut Consumer<T, N>,
        producer: &mut Option<thread::JoinHandle<()>>,
    ) -> T {
        loop {
            if let Some(v) = c.pop() {
                return v;
            }
            match producer {
                Some(h) if h.is_finished() => {
                    // join() synchronizes with the thread's exit, so every push it
                    // made is visible to the next pop.
                    if let Err(panic) = producer.take().unwrap().join() {
                        std::panic::resume_unwind(panic);
                    }
                }
                Some(_) => thread::yield_now(),
                None => panic!("producer exited before sending every item"),
            }
        }
    }

    fn join(producer: Option<thread::JoinHandle<()>>) {
        if let Some(h) = producer {
            h.join().unwrap();
        }
    }

    // ---- single-threaded behaviour ----

    #[test]
    fn fifo_full_and_empty() {
        let (mut p, mut c) = channel::<u32, 4>();
        assert_eq!(c.pop(), None);
        for i in 0..4 {
            p.push(i).unwrap();
        }
        assert_eq!(p.push(99), Err(99));
        for i in 0..4 {
            assert_eq!(c.pop(), Some(i));
        }
        assert_eq!(c.pop(), None);
    }

    #[test]
    fn capacity_one() {
        // MASK == 0: every index maps to the single slot.
        let (mut p, mut c) = channel::<String, 1>();
        for i in 0..10 {
            p.push(i.to_string()).unwrap();
            assert_eq!(p.push("x".into()), Err("x".into()));
            assert_eq!(c.pop().as_deref(), Some(i.to_string().as_str()));
            assert_eq!(c.pop(), None);
        }
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn rejects_non_power_of_two_capacity() {
        let _ = channel::<u8, 3>();
    }

    #[test]
    fn full_push_returns_the_same_value_untouched() {
        let (mut p, _c) = channel::<Box<u64>, 2>();
        p.push(Box::new(1)).unwrap();
        p.push(Box::new(2)).unwrap();
        let b = Box::new(3);
        let addr = &*b as *const u64;
        let back = p.push(b).unwrap_err();
        assert_eq!(&*back as *const u64, addr);
        assert_eq!(*back, 3);
    }

    #[test]
    fn wraps_slot_index_many_times() {
        let (mut p, mut c) = channel::<usize, 4>();
        for i in 0..100 {
            p.push(i).unwrap();
            p.push(i + 1000).unwrap();
            assert_eq!(c.pop(), Some(i));
            assert_eq!(c.pop(), Some(i + 1000));
        }
    }

    #[test]
    fn counters_wrap_past_usize_max() {
        // Start 2 below the wrap with a full ring straddling it: tail wraps to a
        // small number while head is still huge, so full/empty rely on wrapping_sub.
        let (mut p, mut c) = channel_at::<usize, 4>(usize::MAX - 1);
        for i in 0..4 {
            p.push(i).unwrap();
        }
        assert_eq!(p.push(9), Err(9), "full across the wrap");
        for i in 0..4 {
            assert_eq!(c.pop(), Some(i));
            p.push(i + 10).unwrap();
        }
        for i in 0..4 {
            assert_eq!(c.pop(), Some(i + 10));
        }
        assert_eq!(c.pop(), None, "empty across the wrap");
    }

    #[test]
    fn zero_sized_values() {
        let (mut p, mut c) = channel::<(), 4>();
        for _ in 0..4 {
            p.push(()).unwrap();
        }
        assert_eq!(p.push(()), Err(()));
        for _ in 0..4 {
            assert_eq!(c.pop(), Some(()));
        }
        assert_eq!(c.pop(), None);
    }

    // ---- drop accounting ----

    #[test]
    fn drops_unconsumed_items_exactly_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut p, mut c) = channel::<Counted, 8>();
        for id in 0..6 {
            p.push(Counted { id, drops: drops.clone() }).unwrap();
        }
        assert_eq!(c.pop().unwrap().id, 0);
        assert_eq!(c.pop().unwrap().id, 1);
        assert_eq!(drops.load(Ordering::Relaxed), 2);
        drop((p, c)); // the 4 still queued are dropped by Ring::drop
        assert_eq!(drops.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn drop_of_wrapped_ring_drops_only_live_slots() {
        // After 6 pushes and 4 pops, head = usize::MAX (slot 3) and tail has wrapped
        // to 1 (slot 1): the live values sit in slots 3 and 0, straddling both the
        // array end and the counter wrap.
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut p, mut c) = channel_at::<Counted, 4>(usize::MAX - 4);
        for id in 0..6 {
            if id >= 2 {
                drop(c.pop().unwrap());
            }
            p.push(Counted { id, drops: drops.clone() }).unwrap();
        }
        assert_eq!(drops.load(Ordering::Relaxed), 4);
        assert_eq!(c.0.head.0.load(Ordering::Relaxed) & 3, 3);
        assert_eq!(c.0.tail.0.load(Ordering::Relaxed) & 3, 1);
        drop((p, c));
        assert_eq!(drops.load(Ordering::Relaxed), 6);
    }

    #[test]
    fn either_handle_can_outlive_the_other() {
        let drops = Arc::new(AtomicUsize::new(0));

        // Producer gone: consumer still drains what was pushed.
        let (mut p, mut c) = channel::<Counted, 4>();
        for id in 0..3 {
            p.push(Counted { id, drops: drops.clone() }).unwrap();
        }
        drop(p);
        for id in 0..3 {
            assert_eq!(c.pop().unwrap().id, id);
        }
        assert!(c.pop().is_none());
        drop(c);
        assert_eq!(drops.load(Ordering::Relaxed), 3);

        // Consumer gone: producer can still push until full; the ring drops them.
        let (mut p, c) = channel::<Counted, 4>();
        drop(c);
        for id in 0..4 {
            p.push(Counted { id, drops: drops.clone() }).unwrap();
        }
        assert!(p.push(Counted { id: 4, drops: drops.clone() }).is_err()); // dropped here
        assert_eq!(drops.load(Ordering::Relaxed), 4);
        drop(p);
        assert_eq!(drops.load(Ordering::Relaxed), 8);
    }

    // ---- randomized model check against VecDeque ----

    /// Random push/pop sequence checked op-by-op against a VecDeque, with drop
    /// counts checked after every step. Push probability alternates between 3/4
    /// and 1/4 every 32 ops so the ring keeps hitting both full and empty.
    fn model_check<const N: usize>(seed: u64, start: usize) {
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut p, mut c) = channel_at::<Counted, N>(start);
        let mut model = VecDeque::new();
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut created = 0;
        for i in 0..MODEL_OPS {
            let push_odds = if (i / 32) % 2 == 0 { 3 } else { 1 };
            if xorshift(&mut rng) % 4 < push_odds {
                let id = created;
                created += 1;
                match p.push(Counted { id, drops: drops.clone() }) {
                    Ok(()) => {
                        assert!(model.len() < N, "push succeeded on a full ring");
                        model.push_back(id);
                    }
                    Err(back) => {
                        assert_eq!(model.len(), N, "push failed on a non-full ring");
                        assert_eq!(back.id, id);
                    }
                }
            } else {
                assert_eq!(c.pop().map(|v| v.id), model.pop_front());
            }
            assert_eq!(drops.load(Ordering::Relaxed), created - model.len());
        }
        drop((p, c));
        assert_eq!(drops.load(Ordering::Relaxed), created);
    }

    fn model_check_all_starts<const N: usize>() {
        for seed in 0..SEEDS {
            model_check::<N>(seed, 0);
            model_check::<N>(seed, usize::MAX - N - 7); // crosses the wrap
        }
    }

    #[test]
    fn model_n1() {
        model_check_all_starts::<1>();
    }

    #[test]
    fn model_n2() {
        model_check_all_starts::<2>();
    }

    #[test]
    fn model_n8() {
        model_check_all_starts::<8>();
    }

    #[test]
    fn model_n64() {
        model_check_all_starts::<64>();
    }

    // ---- cross-thread ----

    /// Producer streams 0..ITEMS as Boxes; consumer checks order and that none are
    /// lost. A torn or doubled read of a Box shows up as a bad free under Miri.
    fn stream<const N: usize>(start: usize) {
        let (mut p, mut c) = channel_at::<Box<usize>, N>(start);
        let mut producer = Some(thread::spawn(move || {
            for i in 0..ITEMS {
                push_spin(&mut p, Box::new(i));
            }
        }));
        for i in 0..ITEMS {
            assert_eq!(*pop_or_fail(&mut c, &mut producer), i);
        }
        join(producer);
        assert_eq!(c.pop(), None);
    }

    #[test]
    fn cross_thread_n1() {
        stream::<1>(0);
    }

    #[test]
    fn cross_thread_n2() {
        stream::<2>(0);
    }

    #[test]
    fn cross_thread_n16() {
        stream::<16>(0);
    }

    #[test]
    fn cross_thread_across_usize_wrap() {
        stream::<8>(usize::MAX - ITEMS / 2);
    }

    #[test]
    fn cross_thread_large_values_are_not_torn() {
        // 256-byte values take many instructions to copy; every word must agree.
        let (mut p, mut c) = channel::<[usize; 32], 4>();
        let mut producer = Some(thread::spawn(move || {
            for i in 0..ITEMS {
                push_spin(&mut p, [i; 32]);
            }
        }));
        for i in 0..ITEMS {
            assert_eq!(pop_or_fail(&mut c, &mut producer), [i; 32]);
        }
        join(producer);
    }

    #[test]
    fn cross_thread_consumer_quits_early_ring_drops_rest() {
        // Consumer stops N short; the leftovers fit, so the producer finishes, and
        // whichever handle drops last frees them on its own thread.
        const N: usize = 8;
        let drops = Arc::new(AtomicUsize::new(0));
        let (mut p, mut c) = channel::<Counted, N>();
        let d = drops.clone();
        let mut producer = Some(thread::spawn(move || {
            for id in 0..ITEMS {
                push_spin(&mut p, Counted { id, drops: d.clone() });
            }
        }));
        for id in 0..ITEMS - N {
            assert_eq!(pop_or_fail(&mut c, &mut producer).id, id);
        }
        drop(c);
        join(producer);
        assert_eq!(drops.load(Ordering::Relaxed), ITEMS);
    }

    #[test]
    fn cross_thread_zero_sized_values() {
        let (mut p, mut c) = channel::<(), 4>();
        let mut producer = Some(thread::spawn(move || {
            for _ in 0..ITEMS {
                push_spin(&mut p, ());
            }
        }));
        for _ in 0..ITEMS {
            pop_or_fail(&mut c, &mut producer);
        }
        join(producer);
        assert_eq!(c.pop(), None);
    }

    // ---- type-level guarantees ----

    #[test]
    fn handles_are_send_for_send_types() {
        fn assert_send<T: Send>() {}
        assert_send::<Producer<String, 4>>();
        assert_send::<Consumer<String, 4>>();
        // Not Clone, so there is exactly one of each; checked by the absence of
        // a Clone impl (a compile-fail test would need trybuild).
    }
}
