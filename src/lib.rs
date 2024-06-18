use std::collections::{BTreeMap, HashSet};

use crate::CharacterClass::{Ascii, InvalidChars, ValidChars};

#[derive(PartialEq, Eq, Clone, Default, Debug)]
pub struct CharSet {
    pub low_mask: u32,
    pub high_mask: u64,
    pub non_ascii: HashSet<char>,
}

impl CharSet {
    pub fn contains(&self, char: char) -> bool {
        let val = char as u32 - 1;

        if val > 127 {
            self.non_ascii.contains(&char)
        } else if val > 63 {
            let bit = 1 << (val - 64);
            self.high_mask & bit != 0
        } else {
            let bit = 1 << val;
            self.low_mask & bit != 0
        }
    }
}

#[derive(Debug)]
pub struct Params {
    pub map: BTreeMap<String, String>,
}

impl PartialEq for Params {
    fn eq(&self, other: &Self) -> bool {
        self.map == other.map
    }
}

impl Params {
    pub fn new() -> Params {
        Params {
            map: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, key: String, value: String) {
        self.map.insert(key, value);
    }

    pub fn find(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(|s| &s[..])
    }
}

#[derive(Clone, Debug)]
pub struct Thread {
    pub state: usize,
    pub captures: Vec<(usize, usize)>,
    pub capture_begin: Option<usize>,
}

impl Thread {
    pub fn new() -> Self {
        Self {
            state: 0,
            captures: Vec::new(),
            capture_begin: None,
        }
    }

    pub fn start_capture(&mut self, start: usize) {
        self.capture_begin = Some(start);
    }

    pub fn end_capture(&mut self, end: usize) {
        self.captures.push((self.capture_begin.unwrap(), end));
        self.capture_begin = None;
    }

    pub fn extract<'a>(&self, source: &'a str) -> Vec<&'a str> {
        self.captures
            .iter()
            .map(|&(start, end)| &source[start..end])
            .collect()
    }
}

#[derive(PartialEq, Eq, Clone, Debug)]
pub enum CharacterClass {
    Ascii(u64, u64, bool),
    ValidChars(CharSet),
    InvalidChars(CharSet),
}

impl CharacterClass {
    pub fn any() -> CharacterClass {
        Ascii(u64::MAX, u64::MAX, false)
    }

    pub fn valid_char(char: char) -> Self {
        let val = char as u32 - 1;

        if val > 127 {
            ValidChars(Self::char_to_set(char))
        } else if val > 63 {
            Ascii(1 << (val - 64), 0, false)
        } else {
            Ascii(0, 1 << val, false)
        }
    }

    pub fn invalid_char(char: char) -> Self {
        let val = char as u32 - 1;

        if val > 127 {
            InvalidChars(Self::char_to_set(char))
        } else if val > 63 {
            Ascii(u64::MAX ^ (1 << (val - 64)), u64::MAX, true)
        } else {
            Ascii(u64::MAX, u64::MAX ^ (1 << val), true)
        }
    }

    pub fn char_to_set(char: char) -> CharSet {
        let mut set = CharSet::default();
        set.non_ascii.insert(char);
        set
    }

    pub fn matches(&self, char: char) -> bool {
        match *self {
            ValidChars(ref valid) => valid.contains(char),
            InvalidChars(ref valid) => !valid.contains(char),
            Ascii(high, low, unicode) => {
                let val = char as u32 - 1;
                if val > 127 {
                    unicode
                } else if val > 63 {
                    high & (1 << (val - 64)) != 0
                } else {
                    low & (1 << val) != 0
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct Metadata {
    pub statics: u32,
    pub dynamics: u32,
    pub wildcards: u32,
    pub param_names: Vec<String>,
}

impl Metadata {
    pub fn new() -> Metadata {
        Metadata {
            statics: 0,
            dynamics: 0,
            wildcards: 0,
            param_names: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct State<T> {
    pub index: usize,
    pub chars: CharacterClass,
    pub next_states: Vec<usize>,
    pub acceptance: bool,
    pub start_capture: bool,
    pub end_capture: bool,
    pub metadata: Option<T>,
}

#[derive(Debug)]
pub struct NFA<T> {
    pub states: Vec<State<T>>,
    pub start_capture: Vec<bool>,
    pub end_capture: Vec<bool>,
    pub acceptance: Vec<bool>,
}

impl<T> NFA<T> {
    pub fn put(&mut self, index: usize, chars: CharacterClass) -> usize {
        {
            // Check if the state already exists
            // If it does, return just the index of it
            // So we don't have to create a new state
            let state = self.get(index);

            for &index in &state.next_states {
                let state = self.get(index);

                if state.chars == chars {
                    return index;
                }
            }
        }

        // If the state doesn't exist, we create a new one
        // And add it to the next states of the current state
        let state = self.new_state(chars);
        self.get_mut(index).next_states.push(state);

        state
    }

    pub fn get_mut(&mut self, index: usize) -> &mut State<T> {
        &mut self.states[index]
    }

    pub fn new_state(&mut self, chars: CharacterClass) -> usize {
        // The index of the new state is the length of the states vector
        // Example:
        // [0: 'a', 1: 'b', 2: 'c']
        // The index is 3, so the new state will be at index 3
        let index = self.states.len();
        let state = State::new(index, chars);
        self.states.push(state);

        self.acceptance.push(false);
        self.start_capture.push(false);
        self.end_capture.push(false);

        index
    }

    pub fn get(&self, index: usize) -> &State<T> {
        &self.states[index]
    }

    pub fn acceptance(&mut self, index: usize) {
        // Set the acceptance of the state at the given index to true
        self.get_mut(index).acceptance = true;
        self.acceptance[index] = true;
    }

    pub fn metadata(&mut self, index: usize, metadata: T) {
        self.get_mut(index).metadata = Some(metadata);
    }

    pub fn start_capture(&mut self, index: usize) {
        self.get_mut(index).start_capture = true;
        self.start_capture[index] = true;
    }

    pub fn end_capture(&mut self, index: usize) {
        self.get_mut(index).end_capture = true;
        self.end_capture[index] = true;
    }

    pub fn put_state(&mut self, index: usize, child: usize) {
        if !self.get(index).next_states.contains(&child) {
            self.get_mut(index).next_states.push(child);
        }
    }
}

impl<T> State<T> {
    pub fn new(index: usize, chars: CharacterClass) -> Self {
        Self {
            index,
            chars,
            next_states: Vec::new(),
            acceptance: false,
            start_capture: false,
            end_capture: false,
            metadata: None,
        }
    }
}

#[derive(Debug)]
pub struct Match<'a> {
    pub state: usize,
    pub captures: Vec<&'a str>,
}

impl<'a> Match<'a> {
    pub fn new(state: usize, captures: Vec<&'a str>) -> Self {
        Self { state, captures }
    }
}

#[derive(Debug)]
pub struct RouterMatch<T> {
    pub handler: T,
    pub params: Params,
}

impl<T> RouterMatch<T> {
    pub fn new(handler: T, params: Params) -> Self {
        Self { handler, params }
    }
}

impl<T> NFA<T> {
    pub fn new() -> NFA<T> {
        let root = State::new(0, CharacterClass::any());

        NFA {
            states: vec![root],
            start_capture: vec![false],
            end_capture: vec![false],
            acceptance: vec![false],
        }
    }

    pub fn process<'a>(&self, string: &'a str) -> Result<Match<'a>, String> {
        let mut threads = vec![Thread::new()];

        for (i, char) in string.char_indices() {
            let next_threads = self.process_char(threads, char, i);

            if next_threads.is_empty() {
                return Err(format!("No match found for {}", string));
            }

            threads = next_threads;
        }

        let mut returned = threads
            .into_iter()
            .filter(|thread| self.get(thread.state).acceptance);

        let thread = returned.next();

        match thread {
            None => Err(format!("No match found for {}", string)),
            Some(mut thread) => {
                if thread.capture_begin.is_some() {
                    thread.end_capture(string.len());
                }

                let state = self.get(thread.state);
                Ok(Match::new(state.index, thread.extract(string)))
            }
        }
    }

    pub fn process_char(&self, threads: Vec<Thread>, char: char, pos: usize) -> Vec<Thread> {
        let mut returned = Vec::with_capacity(threads.len());

        for mut thread in threads {
            let current_state = self.get(thread.state);

            let mut count = 0;
            let mut found_state = 0;

            for &index in &current_state.next_states {
                let state = &self.get(index);

                if state.chars.matches(char) {
                    count += 1;
                    found_state = index;
                }
            }

            if count == 1 {
                thread.state = found_state;
                capture(self, &mut thread, current_state.index, found_state, pos);
                returned.push(thread);
                continue;
            }

            for &index in &current_state.next_states {
                let state = &self.get(index);

                if state.chars.matches(char) {
                    let mut thread = fork_thread(&thread, state);
                    capture(self, &mut thread, current_state.index, index, pos);
                    returned.push(thread);
                }
            }
        }

        returned
    }
}

fn fork_thread<T>(thread: &Thread, state: &State<T>) -> Thread {
    let mut new_thread = thread.clone();
    new_thread.state = state.index;
    new_thread
}

fn capture<T>(
    nfa: &NFA<T>,
    thread: &mut Thread,
    current_state: usize,
    next_state: usize,
    pos: usize,
) {
    if thread.capture_begin.is_none() && nfa.start_capture[next_state] {
        thread.start_capture(pos);
    }

    if thread.capture_begin.is_some()
        && nfa.end_capture[current_state]
        && next_state > current_state
    {
        thread.end_capture(pos);
    }
}

#[derive(Debug)]
pub struct Router<T> {
    pub nfa: NFA<Metadata>,
    pub handlers: BTreeMap<usize, T>,
}

fn segments(route: &str) -> Vec<(Option<char>, &str)> {
    let predicate = |c| c == '.' || c == '/';

    let mut segments = vec![];
    let mut segment_start = 0;

    while segment_start < route.len() {
        let segment_end = route[segment_start + 1..]
            .find(predicate)
            .map(|i| i + segment_start + 1)
            .unwrap_or_else(|| route.len());
        let potential_sep = route.chars().nth(segment_start);
        let sep_and_segment = match potential_sep {
            Some(sep) if predicate(sep) => (Some(sep), &route[segment_start + 1..segment_end]),
            _ => (None, &route[segment_start..segment_end]),
        };

        segments.push(sep_and_segment);
        segment_start = segment_end;
    }

    segments
}

fn first_byte(s: &str) -> u8 {
    s.as_bytes()[0]
}

impl<T> Router<T> {
    pub fn new() -> Router<T> {
        Router {
            nfa: NFA::new(),
            handlers: BTreeMap::new(),
        }
    }

    pub fn recognize(&self, mut path: &str) -> Result<RouterMatch<&T>, String> {
        if first_byte(path) == b'/' {
            path = &path[1..];
        }

        let nfa = &self.nfa;
        let result = nfa.process(path);

        return result.map(|m| {
            let mut map = Params::new();
            let state = &nfa.get(m.state);
            let metadata = state.metadata.as_ref().unwrap();
            let param_names = metadata.param_names.clone();

            for (i, capture) in m.captures.iter().enumerate() {
                if !param_names[i].is_empty() {
                    map.insert(param_names[i].to_string(), capture.to_string());
                }
            }

            let handler = self.handlers.get(&m.state).unwrap();
            RouterMatch::new(handler, map)
        });
    }

    pub fn add(&mut self, mut route: &str, destiny: T) {
        if route.is_empty() {
            return;
        }

        // Remove leading slash if exists
        if first_byte(route) == b'/' {
            route = &route[1..];
        }

        let nfa = &mut self.nfa;
        let mut state = 0;
        let mut metadata = Metadata::new();

        for (separator, segment) in segments(route) {
            // If we have a separator,
            // we need to add a transition to the current state
            if let Some(separator) = separator {
                state = nfa.put(state, CharacterClass::valid_char(separator));
            }

            if segment.is_empty() {
                continue;
            }

            match first_byte(segment) {
                b':' => {
                    state = process_star_state(nfa, state);
                    metadata.dynamics += 1;
                    metadata.param_names.push(
                        // Add the param key without ':'
                        segment[1..].to_string(),
                    );
                }
                b'*' => {
                    state = process_star_state(nfa, state);
                    metadata.wildcards += 1;
                    metadata.param_names.push(
                        // Add the param key without '*'
                        segment[1..].to_string(),
                    );
                }
                _ => {
                    state = process_static_segment(segment, nfa, state);
                    metadata.statics += 1;
                }
            }
        }

        // Mark the state as an acceptance state
        nfa.acceptance(state);

        // Add the metadata to the state
        nfa.metadata(state, metadata);

        // Add the handler to the handlers map
        self.handlers.insert(state, destiny);
    }
}

fn process_star_state<T>(nfa: &mut NFA<T>, mut state: usize) -> usize {
    state = nfa.put(state, CharacterClass::invalid_char('/'));
    nfa.put_state(state, state);
    nfa.start_capture(state);
    nfa.end_capture(state);

    state
}

fn process_static_segment<T>(segment: &str, nfa: &mut NFA<T>, mut state: usize) -> usize {
    // When we are processing a static segment
    // we just need to add a transition for each character
    // to our current state
    for char in segment.chars() {
        state = nfa.put(state, CharacterClass::valid_char(char));
    }

    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segments() {
        let route = "users/:id";
        let expected = vec![(None, "users"), (Some('/'), ":id")];
        assert_eq!(segments(route), expected);

        let route = "/users/:id/posts";
        let expected = vec![
            (Some('/'), "users"),
            (Some('/'), ":id"),
            (Some('/'), "posts"),
        ];
        assert_eq!(segments(route), expected);

        let route = "/users/:id/posts/:post_id";
        let expected = vec![
            (Some('/'), "users"),
            (Some('/'), ":id"),
            (Some('/'), "posts"),
            (Some('/'), ":post_id"),
        ];
        assert_eq!(segments(route), expected);

        let route = "/users/:id/posts/:post_id/comments";
        let expected = vec![
            (Some('/'), "users"),
            (Some('/'), ":id"),
            (Some('/'), "posts"),
            (Some('/'), ":post_id"),
            (Some('/'), "comments"),
        ];
        assert_eq!(segments(route), expected);
    }

    #[test]
    fn test_add_static_routes() {
        let mut router = Router::new();

        router.add("users", "users");

        let nfa = &router.nfa;
        let handlers = &router.handlers;

        assert_eq!(
            nfa.states.len(),
            6 // One state for each character in "users" + 1 for the root
        );
        assert_eq!(handlers.len(), 1);

        let handler = handlers
            .get(&5) // The last state of the NFA
            .unwrap();

        assert_eq!(*handler, "users");
    }

    #[test]
    fn test_add_routes_with_star_wildcards() {
        let mut router = Router::new();

        router.add("users/*-profile", "users-wildcard");

        let nfa = router.nfa;
        let handlers = router.handlers;

        assert_eq!(handlers.len(), 1);
        assert_eq!(
            nfa.states.len(),
            8 // One state for each character in "users" + 1 for the root
        );
    }

    #[test]
    fn test_add_routes_with_colon_wildcards() {
        let mut router = Router::new();

        router.add("user/:id", "users-wildcard");

        let nfa = router.nfa;
        let handlers = router.handlers;

        assert_eq!(handlers.len(), 1);
        assert_eq!(nfa.states.len(), 7);

        let metadata = &nfa.states[6]; // Last state
        let metadata = metadata.metadata.as_ref().unwrap();
        let params = &metadata.param_names;

        assert_eq!(*params, vec!["id"]);
    }

    #[test]
    fn test_route_recognize_static_route() {
        let mut router = Router::new();

        router.add("/users", "users");

        let m = router.recognize("/users").unwrap();

        assert_eq!(*m.handler, "users");
        assert_eq!(m.params, Params::new());
    }

    #[test]
    fn test_route_recognize_colon_wildcard() {
        let mut router = Router::new();

        router.add("/user/:id", "user");

        let m = router.recognize("/user/1").unwrap();

        assert_eq!(*m.handler, "user");
        assert_eq!(m.params.find("id"), Some("1"));
    }

    #[test]
    fn test_route_recognize_colon_wildcard_multiple_params() {
        let mut router = Router::new();

        router.add("/user/:id/posts/:post_id", "user-post");

        let m = router.recognize("/user/9/posts/10").unwrap();

        assert_eq!(*m.handler, "user-post");
        assert_eq!(m.params.find("id"), Some("9"));
        assert_eq!(m.params.find("post_id"), Some("10"));
    }

    #[test]
    fn test_route_recognize_colon_wildcard_fail() {
        let mut router = Router::new();

        router.add("/user/:id", "user");

        let m = router.recognize("/user");

        assert!(m.is_err());
    }

    #[test]
    fn test_route_recognize_star_wildcard() {
        let mut router = Router::new();

        router.add("/fs/*path", "fs");

        let m = router.recognize("/fs/random-file-path").unwrap();

        assert_eq!(*m.handler, "fs");
        assert_eq!(m.params.find("path"), Some("random-file-path"));
    }
}
