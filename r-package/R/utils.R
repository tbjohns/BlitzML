find_max_lambda = function(x, y, params = c(0, 0, 1, 2)) {
  solver = BlitzML_new_sparse_logreg_solver()
  dataset = BlitzML_new_sparse_dataset(x, y)
  params = BlitzML_new_parameters(params)
  BlitzML_sparse_linear_solver_compute_max_l1_penalty(solver, dataset, params)
}

sigmoid = function(x) {
  1 / (1 + exp(-x))
}

fast_rank = function(...) {
  if(require(data.table))
    data.table::frank(...)
  else
    rank(...)
}

#' @export
auc = function (actual, predicted) {
  actual = as.integer(actual > 0)
  rprob = fast_rank(predicted)
  n1 = sum(actual)
  n0 = length(actual) - n1
  u = sum(rprob[actual == 1]) - n1 * (n1 + 1)/2
  exp(log(u) - log(n1) - log(n0))
}
