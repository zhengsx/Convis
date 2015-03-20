import numpy as np
import pylab as plb
import mpl_toolkits.mplot3d as axes3d
import matplotlib.animation as ani
import matplotlib.colors
import func
import opt
import anim

fig = plb.figure()
ax = axes3d.Axes3D(fig)
surface_interval = 0.25
functype = func.functype("Rosenbrock")
F = functype.func(surface_interval)
ax.plot_surface(F[0], F[1], F[2], rstride=1, cstride=1, cmap='coolwarm', linewidth=0, alpha = 0.5, antialiased=False, norm=matplotlib.colors.LogNorm())
initPoint = (1.0, 8.0)
epoch = 500

# GDlines = opt.)GradientDescent(initPoint, functype,learningrate=0.000001, epoch=epoch)
# # print(GDline)
# gdline = ax.plot(GDlines[0, 0:1], GDlines[1, 0:1], GDlines[2, 0:1], linewidth= 2, color = 'green')[0]
# gdline_ani = ani.FuncAnimation(fig, anim.update_lines, epoch, fargs=(GDlines, gdline), interval=1, blit=False)

# GDMlines = opt.GDMomentum(initPoint, functype, rho=0.8, learningrate=0.000001, epoch=epoch)
# gdmline = ax.plot(GDMlines[0, 0:1], GDMlines[1, 0:1], GDMlines[2, 0:1], linewidth = 2, color = 'yellow')[0]
# gdmline_ani = ani.FuncAnimation(fig, anim.update_lines, epoch, fargs=(GDMlines, gdmline), interval=1, blit=False)

# AGlines = opt.AdaGrad(initPoint, functype, learningrate=1, epoch=epoch)
# agline = ax.plot(AGlines[0, 0:1], AGlines[1, 0:1], AGlines[2, 0:1], linewidth = 2, color = 'gray')[0]
# agline_ani = ani.FuncAnimation(fig, anim.update_lines, epoch, fargs=(AGlines, agline), interval=1, blit=False)

ADlines = opt.AdaDelta(initPoint, functype, rho = 0.95, e = 0.01, epoch=epoch)
# print(ADlines)
adline = ax.plot(ADlines[0, 0:1], ADlines[1, 0:1], ADlines[2, 0:1], linewidth= 2, color = 'blue')[0]
adline_ani = ani.FuncAnimation(fig, anim.update_lines, epoch, fargs=(ADlines, adline), interval=1, blit=False)
adpoint = ax.plot(ADlines[0, 0:1], ADlines[1,0:1], ADlines[2,0:1],linestyle='', marker='o', color='b')[0]
adpoint_ani = ani.FuncAnimation(fig, anim.update_points, epoch, fargs=(ADlines, adpoint), interval=1,blit=False)

NAGlines = opt.NAG(initPoint, functype, rho = 0.95, learningrate = 0.0001, epoch=epoch)
# print(NAGlines)
NAGline = ax.plot(NAGlines[0, 0:1], NAGlines[1, 0:1], NAGlines[2, 0:1], linewidth= 2, color = 'orange')[0]
nagline_ani = ani.FuncAnimation(fig, anim.update_lines, epoch, fargs=(NAGlines, NAGline), interval=1, blit=False)
nagpoint = ax.plot(NAGlines[0, 0:1], NAGlines[1,0:1], NAGlines[2,0:1],linestyle='', marker='o', color='orange')[0]
nagpoint_ani = ani.FuncAnimation(fig, anim.update_points, epoch, fargs=(NAGlines, nagpoint), interval=1,blit=False)


plb.show()
