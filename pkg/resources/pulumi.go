package resources

import "math"

// safeIntToInt32 safely converts an int to int32, clamping to int32 range
func safeIntToInt32(value int) int32 {
	if value < math.MinInt32 {
		return math.MinInt32
	}
	if value > math.MaxInt32 {
		return math.MaxInt32
	}

	return int32(value)
}
