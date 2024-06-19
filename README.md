## Overview

This project serves as a study into the concept of Non-deterministic Finite Automata (NFA) and its applications in route
recognition.
By leveraging the already
existing [route-recognizer](https://github.com/http-rs/route-recognizer/tree/main) library, this project aims to delve
into the
practical implementation and benefits of using NFA for dynamic route handling. The route-recognizer library provides a
solid foundation for exploring these concepts in Rust, demonstrating how NFAs can be used to efficiently match and
manage routes.

**Lightning-Path** is a high-performance route recognizer library for Rust, designed to match URL patterns efficiently
using the Non-deterministic Finite Automaton (NFA) concept. It allows you to define routes and quickly determine which
route matches a given URL path, making it ideal for web frameworks and other applications requiring fast and reliable
routing.

## How to install

Run the following Cargo command in your project directory:

```bash
cargo add lightning-path
```

Or add the following line to your Cargo.toml:

```toml
lightning-path = "1.0.2"
```

## Examples

### Static Routes

```rust
use lightning_path::Router;

fn main() {
    let mut router = Router::new();

    router.add("/home", "Home");
    router.add("/about", "About");
    router.add("/contact", "Contact");

    let m = router.recognize("/home").unwrap();

    assert_eq!(*m.handler, Some("Home"));
}
```

### Dynamic Routes

```rust
use lightning_path::Router;

fn main() {
    let mut router = Router::new();

    router.add("/user/:id", "User");
    router.add("/post/:id", "Post");

    let m = router.recognize("/user/123").unwrap();

    assert_eq!(*m.handler, Some("User"));
    assert_eq!(m.params.find("id"), Some("123"));
}
```

### Wildcard Routes

```rust
use lightning_path::Router;

fn main() {
    let mut router = Router::new();

    router.add("/fs/*path", "fs");

    let m = router.recognize("/fs/random-file-path").unwrap();

    assert_eq!(*m.handler, "fs");
    assert_eq!(m.params.find("path"), Some("random-file-path"));
}
```

