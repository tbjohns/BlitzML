# loss_indices:
# 0 - squared
# 1 - huber
# 2 - logistic

#' @useDynLib "blitzml", .registration=TRUE
#' @importFrom Rcpp sourceCpp
#' @export
get_max_lambda = function(x, y, params = c(l1_penalty = 1, "na" = 0, include_bias_term = 1, loss_index = 2)) {
  solver = BlitzML_new_solver()
  dataset = create_solver_sparse_dataset(x, y)
  params = BlitzML_new_parameters(params)
  BlitzML_sparse_linear_solver_compute_max_l1_penalty(solver, dataset, params)
}
