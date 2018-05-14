# loss_indices:
# 0 - squared
# 1 - huber
# 2 - logistic

#' @useDynLib "blitzml", .registration=TRUE
#' @importFrom Rcpp sourceCpp
#' @export
get_max_lambda = function(x, y, params = c(l1_penalty = 1, "na" = 0, include_bias_term = 1, loss_index = 2)) {
  solver = BlitzML_new_sparse_logreg_solver()
  dataset = BlitzML_new_sparse_dataset(x, y)
  params = BlitzML_new_parameters(params)
  BlitzML_sparse_linear_solver_compute_max_l1_penalty(solver, dataset, params)
}

#' @export
BlitzMLRegression = R6::R6Class(
  "BlitzMLRegression",
  public = list(
    initialize = function(lambda,
                          include_bias_term = TRUE,
                          loss = c("squared", "huber", "logistic"),
                          log_dir = "/dev/null") {
      loss = match.arg(loss)
      loss_index = match(loss, private$supported_losses) - 1L
      private$log_dir = log_dir
      switch (loss,
              logistic = {private$solver = BlitzML_new_sparse_logreg_solver()},
              stop(sprintf("solver for '%s' loss function is not supported yet", loss))
      )
      private$params = BlitzML_new_parameters(c(lambda = lambda,
                         undefined_param = 0,
                         include_bias_term = as.numeric(include_bias_term),
                         loss_index = loss_index))
    },
    fit = function(x, y, init = NULL, ...) {
      stopifnot(is.vector(y))
      stopifnot(inherits(x, "dgCMatrix"))
      stopifnot(nrow(x) == length(y))

      num_features = ncol(x)
      num_examples = length(y)

      result_buffer = numeric(num_features + 1 + num_examples + 2)
      if (!is.null(init))
        result_buffer[1:num_features] = init

      status_buffer = paste(rep(" ", 64), collapse = "")

      dataset = BlitzML_new_sparse_dataset(x, y)

      BlitzML_solve_problem(private$solver, dataset, private$params, result_buffer, status_buffer, private$log_dir)
      return(list(coef = result_buffer[1:num_features],
                  bias = result_buffer[[num_features + 1L]],
                  result = result_buffer,
                  status = status_buffer))
    }
  ),
  private = list(
    supported_losses = c("squared", "huber", "logistic"),
    params = NULL,
    solver = NULL,
    log_dir = NULL
  )
)
