sigmoid = function(x) {
  1 / (1 + exp(-x))
}

#' @export
auc = function (actual, predicted) {
  actual = as.integer(actual > 0)
  rprob = data.table::frank(predicted)
  n1 = sum(actual)
  n0 = length(actual) - n1
  u = sum(rprob[actual == 1]) - n1 * (n1 + 1)/2
  exp(log(u) - log(n1) - log(n0))
}
