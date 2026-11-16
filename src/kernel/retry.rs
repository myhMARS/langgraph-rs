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
#[derive(Clone, Debug, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts, including the first one.
    pub max_attempts: usize,
    /// Backoff before the second attempt.
    pub initial_backoff: Duration,
    /// Multiplier applied after every failed attempt.
    pub multiplier: f64,
    /// Upper bound for the backoff.
    pub max_backoff: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 1,
            initial_backoff: Duration::ZERO,
            multiplier: 1.0,
            max_backoff: Duration::ZERO,
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
        Duration::from_secs_f64(seconds.max(0.0))
    }
}
