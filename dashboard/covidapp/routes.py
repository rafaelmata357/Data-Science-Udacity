from covidapp import app
import json, plotly
from flask import render_template, request, Response, jsonify
from wrangling_scripts.wrangle_data import return_figures

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])
def index():

	# List of countries for filter
	country_codes = ['Costa Rica','Panama','Nicaragua','Guatemala','Honduras','El Salvador','US','Mexico','Canada','Colombia','Chile','Argentina','Venezuela','Ecuador','Uruguay', 'Brazil','Peru','Italy','Spain','France','Germany','United Kingdom']

	# Parse the POST request countries list
	if (request.method == 'POST') and request.form:
		figures = return_figures(request.form)
		countries_selected = []

		for country in request.form.lists():
			countries_selected.append(country)
	
	# GET request returns all countries for initial page load
	else:
		figures = return_figures()
		countries_selected = []
		for country in country_codes:
			countries_selected.append(country)

	# plot ids for the html id tag
	ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

	# Convert the plotly figures to JSON for javascript in html template
	figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)

	return render_template('index.html', ids=ids,
		figuresJSON=figuresJSON,
		all_countries=country_codes,
		countries_selected=countries_selected)
