#' Create a counterparty object
#'
#' @param EAD 
#' @param PD 
#' @param LGD 
#' @param equation 
#'
#' @export
counterparty = function(EAD = EAD, PD = PD, LGD =LGD, equation = equation)
{
  return(new(Counterparty, as.numeric(PD), as.numeric(EAD), as.numeric(LGD), as.numeric(equation)))
}
