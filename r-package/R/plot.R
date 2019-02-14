error.bars = function (x, upper, lower, width = 0.02, ...) {
  xlim <- range(x)
  barw <- diff(xlim) * width
  graphics::segments(x, upper, x, lower, ...)
  graphics::segments(x - barw, upper, x + barw, upper, ...)
  graphics::segments(x - barw, lower, x + barw, lower, ...)
  range(upper, lower)
}

plot.cv.BlitzMLLassoRegression = function (x, ...) {
  cvobj = x
  xlab = "log10(lambda)"

  new.args = list(...)
  if (length(new.args))  plot.args[names(new.args)] = new.args
  plot.args = list(x = log10(cvobj$lambda),
                   y = cvobj$cvm,
                   ylim = range(cvobj$cvup, cvobj$cvlo),
                   xlab = xlab,
                   ylab = "auc",
                   type = "n")
  do.call("plot", plot.args)

  error.bars(log10(cvobj$lambda), cvobj$cvup,
             cvobj$cvlo, width = 0.01, col = "darkgrey")
  graphics::points(log10(cvobj$lambda), cvobj$cvm, pch = 20, col = "red")
  invisible()
}
