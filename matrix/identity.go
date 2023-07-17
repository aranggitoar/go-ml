package matrix

import "fmt"

func (m *Matrix[T]) Shape() (int, int) {
	a := m.Value
	return len(a), len(a[0])
}

func (m *Matrix[T]) SprintfShape() string {
	a := m.Value
	return fmt.Sprintf("(%v, %v)", len(a), len(a[0]))
}

func (m *Matrix[T]) Size() int {
	a := m.Value
	return len(a) * len(a[0])
}
