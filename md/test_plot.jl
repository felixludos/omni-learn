using Plots

plotlyjs()

p = plot(scatter(rand(100), rand(100)))

restyle!(p)

pl = plot(scatter(x=1:10, y=1:10));
pl.data[1].fields

meshgrid
