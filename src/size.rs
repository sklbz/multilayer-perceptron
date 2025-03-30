pub(crate) trait Size {}

impl Size for usize {}
impl Size for (usize, usize) {}
impl Size for (usize, Vec<usize>) {}
impl Size for (usize, usize, usize) {}
impl Size for (usize, Vec<usize>, Vec<usize>) {}
