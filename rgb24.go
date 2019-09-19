package main

import (
	"image"
	"image/color"
)

/*
RGB24 is a structure holding a 24-bit raw RGB image
*/
type RGB24 struct {
	Pix    []uint8
	Stride int
	Rect   image.Rectangle
}

/*
RGB is a single pixel in the RGB24 image type
*/
type RGB struct {
	R, G, B uint8
}

/*
RGBModel is the color model for a 24-bit image
*/
var RGBModel color.Model = color.ModelFunc(func(c color.Color) color.Color {
	r, g, b, _ := c.RGBA()
	return RGB{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8)}
})

/*
RGBA implements the Color interface for the RGB pixel type
*/
func (c RGB) RGBA() (r, g, b, a uint32) {
	r = uint32(c.R) << 8
	g = uint32(c.G) << 8
	b = uint32(c.B) << 8
	return
}

/*
NewRGB creates a black RGB24 image
*/
func NewRGB(r image.Rectangle) *RGB24 {
	return &RGB24{
		Rect:   r.Canon(),
		Stride: 3 * r.Dx(),
		Pix:    make([]uint8, 3*r.Dx()*r.Dy()),
	}
}

/*
FromImage constructs an RGB24 from a given image
*/
func FromImage(img image.Image) *RGB24 {
	if r, ok := img.(*RGB24); ok {
		return r
	}
	// this is really slow for now...
	r := NewRGB(img.Bounds())
	for x := r.Rect.Min.X; x < r.Rect.Max.X; x++ {
		for y := r.Rect.Min.Y; y < r.Rect.Max.Y; y++ {
			r.Set(x, y, img.At(x, y))
		}
	}
	return r
}

/*
FromRaw constructs an RGB24 image from raw bytes in RGB-row major order
*/
func FromRaw(b []byte, stride int, cols int, rows int) *RGB24 {

	ret := &RGB24{
		Pix:    b,
		Stride: stride,
		Rect:   image.Rect(0, 0, cols, rows),
	}
	return ret
}

/*
At implements the image.Image interface for RGB24
*/
func (p *RGB24) At(x, y int) color.Color {
	if !(image.Point{x, y}.In(p.Rect)) {
		return RGB{}
	}
	i := p.PixOffset(x, y)
	return RGB{
		p.Pix[i], p.Pix[i+1], p.Pix[i+2],
	}
}

/*
Set implements the image.Image interface for RGB24
*/
func (p *RGB24) Set(x, y int, c color.Color) {
	if !(image.Point{x, y}.In(p.Rect)) {
		return
	}
	i := p.PixOffset(x, y)
	c1 := RGBModel.Convert(c).(RGB)
	p.Pix[i+0] = uint8(c1.R)
	p.Pix[i+1] = uint8(c1.G)
	p.Pix[i+2] = uint8(c1.B)
}

/*
ColorModel implements the image.Image interface for RGB24
*/
func (p *RGB24) ColorModel() color.Model {
	return RGBModel
}

/*
SubImage returns an *RGB24 which is the crop of the given image using the rectangle r
*/
func (p *RGB24) SubImage(r image.Rectangle) image.Image {
	r = r.Intersect(p.Rect)
	// If r1 and r2 are Rectangles, r1.Intersect(r2) is not guaranteed to be inside
	// either r1 or r2 if the intersection is empty. Without explicitly checking for
	// this, the Pix[i:] expression below can panic.
	if r.Empty() {
		return &RGB24{}
	}
	// TODO: implement this much faster sub image routine, but this requires image stride
	//   in the C code.  right now just copy the image bytes to a new slice
	// i := p.PixOffset(r.Min.X, r.Min.Y)
	// return &RGB24{
	// 	Pix:    p.Pix[i:],
	// 	Stride: p.Stride,
	// 	Rect:   r,
	// }
	ret := &RGB24{
		Stride: r.Dx() * 3,
		Rect:   image.Rect(0, 0, r.Dx(), r.Dy()),
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			i := p.PixOffset(x, y)
			ret.Pix = append(ret.Pix, p.Pix[i], p.Pix[i+1], p.Pix[i+2])
		}
	}
	return ret
}

// PixOffset returns the index of the first element of Pix that corresponds to
// the pixel at (x, y).
func (p *RGB24) PixOffset(x, y int) int {
	return (y-p.Rect.Min.Y)*p.Stride + (x-p.Rect.Min.X)*3
}

/*
Bounds returns the bounding rectangle of the image
*/
func (p *RGB24) Bounds() image.Rectangle {
	return p.Rect
}
