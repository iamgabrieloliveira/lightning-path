use std::collections::{BTreeMap, HashSet};

use crate::CharacterClass::{Ascii, ValidChars};

#[derive(PartialEq, Eq, Clone, Default, Debug)]
struct CharSet {
    low_mask: u32,
    high_mask: u64,
    non_ascii: HashSet<char>,
}

#[derive(PartialEq, Eq, Clone, Debug)]
enum CharacterClass {
    Ascii(u64, u64, bool),
    ValidChars(CharSet),
    InvalidChars(CharSet),
}

impl CharacterClass {
    fn any() -> CharacterClass {
        Ascii(u64::MAX, u64::MAX, false)
    }

    fn valid_char(char: char) -> Self {
        let val = char as u32 - 1;

        if val > 127 {
            ValidChars(Self::char_to_set(char))
        } else if val > 63 {
            Ascii(1 << (val - 64), 0, false)
        } else {
            Ascii(0, 1 << val, false)
        }
    }

    fn char_to_set(char: char) -> CharSet {
        let mut set = CharSet::default();
        set.non_ascii.insert(char);
        set
    }
}

#[derive(Debug)]
struct Metadata {
    statics: u32,
    dynamics: u32,
    wildcards: u32,
    param_names: Vec<String>,
}

impl Metadata {
    fn new() -> Metadata {
        Metadata {
            statics: 0,
            dynamics: 0,
            wildcards: 0,
            param_names: Vec::new(),
        }
    }
}

#[derive(Debug)]
struct State<T> {
    index: usize,
    chars: CharacterClass,
    next_states: Vec<usize>,
    acceptance: bool,
    start_capture: bool,
    end_capture: bool,
    metadata: Option<T>,
}

#[derive(Debug)]
struct NFA<T> {
    states: Vec<State<T>>,
    start_capture: Vec<bool>,
    end_capture: Vec<bool>,
    acceptance: Vec<bool>,
}

impl<T> NFA<T> {
    fn put(&mut self, index: usize, chars: CharacterClass) -> usize {
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

    fn get_mut(&mut self, index: usize) -> &mut State<T> {
        &mut self.states[index]
    }

    fn new_state(&mut self, chars: CharacterClass) -> usize {
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

    fn get(&self, index: usize) -> &State<T> {
        &self.states[index]
    }

    fn acceptance(&mut self, index: usize) {
        // Set the acceptance of the state at the given index to true
        self.get_mut(index).acceptance = true;
        self.acceptance[index] = true;
    }

    fn metadata(&mut self, index: usize, metadata: T) {
        self.get_mut(index).metadata = Some(metadata);
    }
}

impl<T> State<T> {
    fn new(index: usize, chars: CharacterClass) -> Self {
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

impl<T> NFA<T> {
    fn new() -> NFA<T> {
        let root = State::new(0, CharacterClass::any());

        NFA {
            states: vec![root],
            start_capture: vec![false],
            end_capture: vec![false],
            acceptance: vec![false],
        }
    }
}

#[derive(Debug)]
struct Router<T> {
    nfa: NFA<Metadata>,
    handlers: BTreeMap<usize, T>,
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

impl<T: std::fmt::Debug> Router<T> {
    fn new() -> Router<T> {
        Router {
            nfa: NFA::new(),
            handlers: BTreeMap::new(),
        }
    }

    fn add(&mut self, mut route: &str, destiny: T) {
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
                b':' => todo!(),
                b'*' => todo!(),
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

    fn process_static_segment(&mut self, segment: &str, mut state: usize) -> usize {
        for char in segment.chars() {
            state = self.nfa.put(state, CharacterClass::valid_char(char));
        }

        state
    }
}

fn process_static_segment<T>(segment: &str, nfa: &mut NFA<T>, mut state: usize) -> usize {
    for char in segment.chars() {
        state = nfa.put(state, CharacterClass::valid_char(char));
    }

    state
}

fn main() {}
