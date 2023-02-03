def year_vector(start_year,stop_year):
	"""
	* Makes an array of consecutive year numbers from given start
	  year to given stop year.  

	 CALLS FUNCTIONS:

	 INPUT:
	* start_year (int): starting year of array.
	* stop_year (int): last year of array.
	 
	 OUTPUT:
	* years (array of strings)
	"""
	years = ['{}'.format(i) for i in range(start_year,stop_year+1)]
	return years