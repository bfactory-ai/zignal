//! Feature detection and description module for Zignal
//!
//! This module provides ORB (Oriented FAST and Rotated BRIEF) feature detection,
//! which is a fast, patent-free alternative to SIFT and SURF suitable for
//! real-time applications.

// Core data structures
pub const KeyPoint = @import("features/keypoint.zig").KeyPoint;
pub const BinaryDescriptor = @import("features/descriptor.zig").BinaryDescriptor;

// Feature detection
pub const Fast = @import("features/fast.zig").Fast;
pub const Orb = @import("features/orb.zig").Orb;

// Feature matching
pub const BruteForceMatcher = @import("features/matcher.zig").BruteForceMatcher;
pub const Match = @import("features/matcher.zig").Match;
pub const MatchStats = @import("features/matcher.zig").MatchStats;

// Re-export useful functions
pub const filterMatchesRatioTest = @import("features/matcher.zig").filterMatchesRatioTest;

test {
    // Run all feature module tests
    _ = @import("features/keypoint.zig");
    _ = @import("features/descriptor.zig");
    _ = @import("features/fast.zig");
    _ = @import("features/orb.zig");
    _ = @import("features/matcher.zig");
}
