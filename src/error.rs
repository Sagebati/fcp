/// Early-return with a custom error. Like `anyhow::bail!` but for typed errors.
///
/// ```ignore
/// throw!(MyError::NotFound);
/// throw!(MyError::InvalidInput(format!("bad: {x}")));
/// ```
#[macro_export]
macro_rules! throw {
    ($err:expr) => {
        return ::core::result::Result::Err($err.into())
    };
}

/// Assert a condition, returning a custom error if it fails.
/// Like `anyhow::ensure!` but for typed errors.
///
/// ```ignore
/// guard!(!items.is_empty(), MyError::Empty);
/// guard!(x > 0, MyError::InvalidValue(x));
/// ```
#[macro_export]
macro_rules! guard {
    ($cond:expr, $err:expr) => {
        if !$cond {
            $crate::throw!($err);
        }
    };
}

/// Pattern-match on the `Err` variant of a `Result`.
/// The `Ok` value passes through; error arms must produce the same type.
///
/// ```ignore
/// let val = catch!(fallible(), {
///     MyError::NotFound => default_value,
///     MyError::Timeout(d) => {
///         warn!("timed out after {d:?}");
///         fallback
///     }
/// });
/// ```
#[macro_export]
macro_rules! catch {
    ($result:expr, { $($arm:pat => $handler:expr),* $(,)? }) => {
        match $result {
            ::core::result::Result::Ok(v) => v,
            ::core::result::Result::Err(e) => match e {
                $($arm => $handler,)*
            },
        }
    };
}
