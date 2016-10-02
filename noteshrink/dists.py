import numpy as np

colors = [
    ('white', (238, 238, 242)),
    ('black', (71, 73, 71)),
    ('gray', (160, 168, 166)),
    ('red', (219, 83, 86)),
    ('pink', (243, 179, 182)),
]

def sv(c):
    cmax = max(c)
    cmin = min(c)
    delta = cmax-cmin
    if cmax == 0:
        s = 0
    else:
        s = float(delta)/cmax
    return s, cmax/255.0

def dist(a, b):
    
    return np.linalg.norm(
        np.array(a, dtype=float)-
        np.array(b, dtype=float))


def hexstr(x):
    r,g,b = x
    i = (r << 16) | (g << 8) | b
    return '{:06x}'.format(i)

print 'RGB values:'
for name, rgb in colors:
    print ' ', name, rgb
print

print 'hex values:'
for name, rgb in colors:
    print ' ', name, hexstr(rgb)
print
    
print 'Euclidean dist. from BG:'
for name, rgb in colors:
    print ' ', name, dist(rgb, colors[0][1])
print


        
sw, vw = sv(colors[0][1])

print 'Value and saturation:'
for name, rgb in colors:
    s,v = sv(rgb)
    print '| **{}** | {:.3f} | {:.3f} | **{:.3f}** | **{:.3f}**'.format(
        name, v, s, abs(v-vw), abs(s-sw))

#print 'white-gray', dist(white,gray)
#print 'white-pink', dist(white,pink)
#print 'white-blue', dist(white,blue)
