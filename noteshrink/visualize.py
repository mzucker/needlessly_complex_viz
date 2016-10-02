#!/usr/bin/env python

'''Converts sequence of images to compact PDF while removing speckles,
bleedthrough, etc.

'''

# for some reason pylint complains about members being undefined :(
# pylint: disable=E1101

import json

from noteshrink import *

def reduce_and_apply(samples, fg_mask, palette, max_samples=1000):

    fg_palette = palette[1:]
    fg_samples = samples[fg_mask]

    num_samples = len(fg_samples)
    num_samples = min(num_samples, max_samples)
    fg_samples = fg_samples[:num_samples]

    print '  applying pallete to {} samples...'.format(num_samples)
    
    fg_labels, _ = vq(fg_samples, fg_palette)
    
    return fg_samples, fg_palette, fg_labels

def get_segments(num_centers, fg_labels, positions):

    segments = []
    radii = []

    for i in range(num_centers):

        pos_indices = np.nonzero(fg_labels==i)[0] + num_centers

        deltas = positions[pos_indices] - positions[i]
        dists = (deltas**2).sum(axis=1)
        
        radii.append(np.sqrt(dists.max()) + 0.03)
        
        pos_indices = pos_indices.reshape( (-1, 1) )
        
        segment_indices = np.hstack((np.ones_like(pos_indices)*i,
                                     pos_indices))
        
        segments.append( [int(idx) for idx in segment_indices.flatten()] )

    return segments, radii

def dump_svg(filename, colors, positions, segments, radii, size=400):

    ostr = open(filename, 'w')

    num_centers = len(segments)
    num_samples = len(positions) - num_centers

    positions = positions * (1, -1, 1)
    positions = positions * size/2 + size/2
    radii = np.array(radii) * size/2

    center_depths = positions[:num_centers,2]
    center_indices = np.argsort(center_depths)

    colors = pack_rgb(colors*255)

    ostr.write('<svg xmlns="http://www.w3.org/2000/svg" '
               'width="{}" height="{}">\n'.format(size, size))

    ostr.write('  <g stroke-width="3">\n')

    for i in center_indices:
        
        cx,cy = positions[i,:2]
        r = radii[i]
        color = colors[i]
        
        ostr.write('    <circle cx="{:.1f}" cy="{:.1f}" '
                   'r="{:.1f}" fill="none" stroke="#{:06x}" '
                   'opacity="0.25"/>\n'.format(cx, cy, r, color))

    ostr.write('  </g>\n')

    for i in center_indices:
        
        segment = np.array(segments[i]).reshape((-1, 2))
        color = colors[i]
        cx,cy = positions[i,:2]
        
        ostr.write('  <g stroke-width="1" stroke="#{:06x}" '
                   'transform="translate({:.1f},{:.1f})">\n'.format(
                       color, cx, cy))
        
        for k in segment[:,1]:
            x2,y2 = positions[k,:2] - (cx,cy)
            ostr.write('    <line x2="{:.1f}" y2="{:.1f}"/>\n'.format(x2,y2))
        ostr.write('  </g>\n')


    point_indices = positions[num_centers:,2].argsort()

    ostr.write('  <g stroke-width="1" stroke="#000000">\n')

    for ii in point_indices:

        i = ii+num_centers
        
        cx,cy = positions[i,:2]
        color = colors[i]

        ostr.write('  <circle cx="{:.1f}" cy="{:.1f}" r="3" '
                   'fill="#{:06x}"/>\n'.format(cx, cy, color))

    ostr.write('  </g>')

    ostr.write('</svg>\n')

    print '  wrote', filename

def dump_json(filename, colors, positions, num_centers, num_samples,
              segments, radii):

    output = dict(
        colors=[round(float(c), 3) for c in colors.flatten()],
        positions=[round(float(p), 3) for p in positions.flatten()],
        points_start=num_centers,
        points_count=num_samples,
        segments=segments,
        radii=[round(float(r), 3) for r in radii])

    ostr = open(filename, 'w')

    json.dump(output, ostr, indent=None)

    print '  wrote', filename
    

def assemble_swatches(filename, pixels, cols=None, zoom=16):

    pixels = pixels.astype(np.uint8)
    
    num_pixels = pixels.shape[0]
    if cols is None:
        cols = int(np.sqrt(num_pixels))
    rows = num_pixels / cols
    size = rows * cols
    if size < num_pixels:
        rows += 1
        size += cols
        pixels = np.vstack((pixels,
                            np.zeros((size-num_pixels, 3),
                                     dtype=pixels.dtype)))
    
    img = pixels[:size].reshape((rows, cols, 3))

    pattern = np.ones((zoom,zoom), dtype=np.uint8)
    
    img = np.kron(img, pattern[:,:,None])

    pil_img = Image.fromarray(img, 'RGB')
    pil_img.save(filename)

    print '  wrote', filename


def dump_samples(basename, samples, max_length):

    num_samples = min(len(samples), max_length)
    samples = samples[:num_samples]

    assemble_swatches(basename + '_samples_raw.png', samples, zoom=3)

    brightness = samples.astype(int).sum(axis=1)

    assemble_swatches(basename + '_samples_sorted.png',
                      samples[brightness.argsort()], zoom=3)

def apply_svd(fg_samples, fg_palette):

    print '  running SVD...'

    colors = np.vstack((fg_palette, fg_samples)).astype(np.float32)

    positions = colors - colors.mean(axis=0)

    colors /= 255.0

    _, _, svd_v = np.linalg.svd(positions,
                                full_matrices=False)

    svd_v[0] *= np.sign(np.mean(svd_v[0]))
    svd_v[1] *= np.sign(np.mean(svd_v[1]))
    svd_v[2] *= np.sign(np.mean(svd_v[2]))

    positions = np.dot(positions, svd_v.T)

    pmin = positions.min(axis=0)
    pmax = positions.max(axis=0)
    prng = pmax-pmin
    pmid = 0.5*(pmax+pmin)

    positions -= pmid
    positions /= prng.max()*0.5
    positions /= 1.1

    return colors, positions

def test_quantize(samples, max_samples=10000):

    samples = samples[:min(len(samples), max_samples)]

    for bits_per_channel in [8, 7, 6, 5, 4]:
        shift = 8 - bits_per_channel
        packed = pack_rgb((samples >> shift) << shift)
        unique, counts = np.unique(packed, return_counts=True)
        mode_index = counts.argmax()
        mode_count = counts[mode_index]
        mode_rgb = unpack_rgb(unique[mode_index])
        print '    {:,d}/{:,d} unique at {} bits per channel'.format(
            len(unique), len(samples), bits_per_channel),
        print '; mode accounts for {:,d} ({:.1f}% of total)'.format(
            mode_count, mode_count*100.0/len(samples))
        print '    bg color is', mode_rgb
    

######################################################################

def visualize():


    parser = get_argument_parser()

    options = parser.parse_args()

    np.random.seed(123456)

    for filename in options.filenames:

        basename, _ = os.path.splitext(os.path.basename(filename))

        img, _ = load(filename)
        print 'opened', filename

        samples = sample_pixels(img, options)

        print '  got {} samples'.format(len(samples))

        test_quantize(samples)
        
        max_swatches = 100**2

        dump_samples(basename, samples, max_swatches)

        palette, fg_mask = get_palette(samples, options, return_mask=True)

        assemble_swatches(basename + '_palette.png', palette, 8, zoom=50)

        modified_palette = palette.copy().astype(np.float32)

        pmin = modified_palette.min()
        pmax = modified_palette.max()

        modified_palette = 255*(modified_palette-pmin)/(pmax-pmin)

        modified_palette = modified_palette.astype(np.uint8)

        assemble_swatches(basename + '_modified_palette.png', modified_palette, 8, zoom=50)
        
        dump_samples(basename + '_fg', samples[fg_mask], max_swatches)
        
        fg_samples, fg_palette, fg_labels = reduce_and_apply(samples, fg_mask,
                                                             palette, 1000)

        num_samples = len(fg_samples)
        num_centers = len(fg_palette)

        colors, positions = apply_svd(fg_samples, fg_palette)

        assert len(colors) == num_samples + num_centers

        segments, radii = get_segments(num_centers, fg_labels, positions)

        dump_json(basename+'_points.json',
                  colors, positions,
                  num_centers, num_samples,
                  segments, radii)

        dump_svg(basename+'_plot.svg',
                 colors, positions, segments, radii)

        print '  done'
        print

if __name__ == '__main__':

    visualize()





    


#make_scatterplot(fg_samples, palette)



