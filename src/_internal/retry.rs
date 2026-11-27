//! Retry policy for node executions.

use std::time::Duration;

/// Retry policy applied to a node.
///
/// `max_attempts` counts the initial execution plus retries: `RetryPolicy::new(3)`
/// runs a failing node at most three times. Backoff is exponential between attempts
/// and capped by `max_backoff`; it is applied with the platform sleep, so retries do
/// not block other superstep nodes longer than necessary (parallel supersteps already
/// joined their threads by then).
///
/// ```
/// use std::time::Duration;
/// use langgraph_rs::RetryPolicy;
///
/// let policy = RetryPolicy::new(3)
///     .with_backoff(Duration::from_millis(10))
///     .with_multiplier(2.0)
///     .with_max_backoff(Duration::from_millis(100));
/// assert_eq!(policy.max_attempts, 3);
/// assert_eq!(policy.backoff_for(1), Duration::from_millis(10));
/// assert_eq!(policy.backoff_for(2), Duration::from_millis(20));
/// ```
#[derive(Clone)]
pub struct RetryPolicy {
    /// Total attempts, including the first one.
    pub max_attempts: usize,
    /// Backoff before the second attempt.
    pub initial_backoff: Duration,
    /// Multiplier applied after every failed attempt.
    pub multiplier: f64,
    /// Upper bound for the backoff.
    pub max_backoff: Duration,
    /// Fraction of the backoff that may be removed deterministically (0.0..=1.0).
    pub jitter: f64,
    /// Only retry when the error message contains one of these substrings (empty = any).
    pub retry_on: Vec<String>,
    /// Predicate form of `retry_on`: LangGraph also accepts a callable receiving the exception.
    /// Adapters supply it; the engine only ever calls back with the error message.
    pub condition: Option<RetryCondition>,
}

/// Callable form of `retry_on`, supplied by an adapter.
pub type RetryCondition = std::sync::Arc<dyn Fn(&str) -> bool + Send + Sync>;

impl std::fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_attempts", &self.max_attempts)
            .field("initial_backoff", &self.initial_backoff)
            .field("multiplier", &self.multiplier)
            .field("max_backoff", &self.max_backoff)
            .field("jitter", &self.jitter)
            .field("retry_on", &self.retry_on)
            .field("condition", &self.condition.is_some())
            .finish()
    }
}

impl PartialEq for RetryPolicy {
    /// The callable predicate has no observable identity, so only its presence is compared.
    fn eq(&self, other: &Self) -> bool {
        self.max_attempts == other.max_attempts
            && self.initial_backoff == other.initial_backoff
            && self.multiplier == other.multiplier
            && self.max_backoff == other.max_backoff
            && self.jitter == other.jitter
            && self.retry_on == other.retry_on
            && self.condition.is_some() == other.condition.is_some()
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff: Duration::ZERO,
            multiplier: 1.0,
            max_backoff: Duration::ZERO,
            jitter: 0.0,
            retry_on: Vec::new(),
            condition: None,
        }
    }
}

impl RetryPolicy {
    /// A policy that tries `max_attempts` times without sleeping between attempts.
    pub fn new(max_attempts: usize) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            ..Self::default()
        }
    }

    /// Set the backoff before the second attempt.
    pub fn with_backoff(mut self, backoff: Duration) -> Self {
        self.initial_backoff = backoff;
        self
    }

    /// Set the exponential multiplier.
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier;
        self
    }

    /// Set an upper bound for the backoff.
    pub fn with_max_backoff(mut self, max_backoff: Duration) -> Self {
        self.max_backoff = max_backoff;
        self
    }

    /// Apply deterministic jitter (a fraction in `0.0..=1.0` removed from the backoff).
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Only retry when the error message contains one of `predicates`.
    pub fn with_retry_on<I, T>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<String>,
    {
        self.retry_on = predicates.into_iter().map(Into::into).collect();
        self
    }

    /// Returns `true` when `message` matches the configured `retry_on` predicates.
    ///
    /// With no predicates configured this applies LangGraph's `default_retry_on`: connection
    /// problems are retried, ordinary programming errors are not, anything else is.
    pub fn should_retry(&self, message: &str) -> bool {
        if let Some(condition) = &self.condition {
            return condition(message);
        }
        if self.retry_on.is_empty() {
            return default_retry_on(message);
        }
        self.retry_on
            .iter()
            .any(|predicate| message.contains(predicate))
    }

    /// Returns `true` when the policy allows more than one attempt.
    pub fn is_retrying(&self) -> bool {
        self.max_attempts > 1
    }

    /// Backoff to wait after the `failed_attempt`-th failure (1-based).
    pub fn backoff_for(&self, failed_attempt: usize) -> Duration {
        if failed_attempt == 0 {
            return Duration::ZERO;
        }
        let mut seconds = self.initial_backoff.as_secs_f64();
        for _ in 1..failed_attempt {
            seconds *= self.multiplier.max(1.0);
        }
        if !self.max_backoff.is_zero() {
            seconds = seconds.min(self.max_backoff.as_secs_f64());
        }
        if self.jitter > 0.0 {
            let fraction = (failed_attempt as f64 * 0.618_033_988_749_895).fract();
            seconds *= 1.0 - self.jitter * fraction;
        }
        Duration::from_secs_f64(seconds.max(0.0))
    }
}

/// LangGraph's `default_retry_on`, matched against the error text.
///
/// `ConnectionError` retries; the usual programming errors do not; everything else does.
fn default_retry_on(message: &str) -> bool {
    if message.contains("ConnectionError") {
        return true;
    }
    const NOT_RETRYABLE: [&str; 11] = [
        "ValueError",
        "TypeError",
        "ArithmeticError",
        "ImportError",
        "LookupError",
        "NameError",
        "SyntaxError",
        "RuntimeError",
        "ReferenceError",
        "StopIteration",
        "OSError",
    ];
    !NOT_RETRYABLE
        .iter()
        .any(|name| message.contains(name))
}
