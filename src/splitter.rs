pub struct TextSplitter<'a> {
    source: &'a [char],
    context_length: usize, // 128 * 4 -> where 4 is the number of chars per token
    splitter: Option<&'a str>,
}

impl<'a> TextSplitter<'a> {
    pub fn new(source: &'a [char], context_length: usize, splitter: Option<&'a str>) -> Self {
        TextSplitter {
            source,
            context_length,
            splitter,
        }
    }
}

impl<'a> Iterator for TextSplitter<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut sl = String::new();
        loop {
            if self.source.len() == 0 {
                return None;
            }
            sl.push(self.source[0]);
            self.source = &self.source[1..];
            if self.splitter.is_some() && sl.ends_with(self.splitter.unwrap()) {
                sl = sl.trim_end_matches("\n").to_string();
                break;
            }
            if sl.len() > self.context_length && !sl.trim().is_empty() {
                break;
            }
        }
        Some(String::from(sl))
    }
}
