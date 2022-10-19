/* Base class to be used as mixin for any component showing plots as a result
of a response Object sent from the server */
var PlotsDiv = {
	mixins: [DataDownloader],
	props: {
		data: {type: Object, default: () => { return{} }},
		// this is used to calculate plot areas and set it in the default layout
		// (use computed body font-size. Note that parseInt('16px')=16):
		plotfontsize: {
			type: Number,
			default: parseInt(window.getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('font-size'))
		},
		downloadUrl: String // base url for download actions
	},
	data(){
		return {
			visible: false, // see watcher below
			// boolean visualizing a div while drawing (to prevent user clicking everywhere for long taks):
			drawingPlots: true,
			// store watchers to dynamically create / remove to avoid useless plot redrawing or calculations (see init)
			watchers: {},
			// an Array of [legendgroup:str, traceProperties:Object] elements:
			legend: [],
			// plots is the data constructed from the received response.
			// It is an Array of Objects, each Object representing a plot P:
			// P: {
			//   traces: Array of Plotly Objects,
			//   params: Object of keys (string) mapped to a value, with the values
			//		   of this plot used to retrieve it when selecting what to show
			//   xaxis: Plotly Object with the xaxis data
			//   yaxis: Plotly Object with the yaxis data
			// }
			plots: [],
			// An Array of Param Objects for laying out plots. Each param has at least the function indexOf(plot, idx, plots)
			// and optionally a value:Array key and a name:str key. If the latter are provided, then the param
			// is displayable as label on the plot grids
			params: [],
			// dict of subplots layout name (string) mapped to an Array of two params dictating the layout.
			// Params are those implemented in params (se above) or other dummy params created in `setoSelection`
			gridlayouts: {},
			// string denoting the selected layout name of gridlayouts (see above)
			selectedgridlayout: '',
			axisOptions: {
				// reminder: x.log and y.log determine the type of axis. Plotly has xaxis.type that can be:
				// ['-', 'linear', 'log', ... other values ], we will set here only 'normal' (log checkbox unselected)
				// or log (log checkbox selected)
				x: {
					log: {disabled: false, value: undefined},
					sameRange: {disabled: false, value: undefined},
					grid: {disabled: false, value: false}
				},
				y: {
					log: {disabled: false, value: undefined},
					sameRange: {disabled: false, value: undefined},
					grid: {disabled: false, value: false}
				}
			},
			// the wait bar while drawing plots
			waitbar: {
				msg: '',  // the message to be displayed, and below some defaults:
				DRAWING: 'Drawing plots... <i class="fa fa-hourglass-o"></i>',
				UPDATING: 'Updating plots... <i class="fa fa-hourglass-o"></i>'
			}
		}
	},
	created(){
		// setup non reactive data:

		// space reserved for the params grid ticklabels:
		this.paramsGridMargin = 3*this.plotfontsize;

		// default Plotly layout. See this.configureLayout for details
		this.defaultlayout = {
			autosize: true,  // without this, the inner svg does not expand properly
			paper_bgcolor: 'rgba(0,0,0,0)', //window.getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('background-color'),
			// font: {family: "Encode Sans Condensed, sans-serif", size: 12}, // this will be overridden
			showlegend: false,
			legend: { bgcolor: 'rgba(0,0,0,0)'},
			margin: {r: 0, b: 0, t: 0, l:0, pad:0},
			annotations: []  // base annotations, it might be enhanced with custom ones (e.g., axis labels)
		};

		// options of the side panel to configure mouse interactions on the plots:
		this.mouseMode = { // https://plot.ly/python/reference/#layout-hovermode
			// hovermodes (plotly keys). Note that we remove the 'y' key because useless
			hovermodes: ["closest", "x", false],
			// the labels of hovermodes to be displayed. Copied from plotly modebar after visual test
			// (note that we remove  the value associated to 'y' because plotly does not implement it
			// and anyway, even providing a mapped label such as 'show y', tests revealed the mode was useless):
			hovermodeLabels: {closest: 'show closest point', x: 'compare data',false: 'do nothing'},
			dragmodes: ["zoom", "pan"],  // "select", "lasso" are useless. false does not seem to work (it's zoom)
			dragmodeLabels: {zoom: 'zoom', pan: 'pan'},
			hovermode: 'closest',  // will set this value to the Plotly layout before plotting, if not explicitly set
			dragmode: 'zoom'  // will set this value to the Plotly layout before plotting, if not explicitly set
		};

		// the plotly config for plots. See
		// https://community.plot.ly/t/remove-options-from-the-hover-toolbar/130/14
		this.defaultplotlyconfig = {
			responsive: true,
			modeBarButtonsToRemove: ['sendDataToCloud', 'toImage'],
			displaylogo: false
		};

		// default layout axis props. https://plotly.com/javascript/reference/layout/xaxis/#layout-xaxis
		// Note that domain and anchor props will be overridden
		this.defaultxaxis = { mirror: true, zeroline: false, linewidth: 1 };
		this.defaultyaxis = { mirror: true, zeroline: false, linewidth: 1 };

		this.colors = {
			_i: -1,
			_values: [
				'#1f77b4',  // muted blue
				'#ff7f0e',  // safety orange
				'#2ca02c',  // cooked asparagus green
				'#d62728',  // brick red
				'#9467bd',  // muted purple
				'#8c564b',  // chestnut brown
				'#e377c2',  // raspberry yogurt pink
				'#7f7f7f',  // middle gray
				'#bcbd22',  // curry yellow-green
				'#17becf'   // blue-teal
			],
			_cmap: {},
			get(key){  // return a new color mapped to key. Subsequent calls with `key` as argument return the same color
				if (!(key in this._cmap)){
					this._cmap[key] = this._values[(++this._i) % this._values.length];
				}
				return this._cmap[key];
			},
			rgba(hexcolor, alpha) {
				// Returns the corresponding 'rgba' string of `hexcolor` with the given alpha channel ( in [0, 1], 1:opaque)
				if (hexcolor.length == 4){
					var [r, g, b] = [hexcolor.substring(1, 2), hexcolor.substring(2, 3), hexcolor.substring(3, 4)];
					var [r, g, b] = [r+r, g+g, b+b];
				}else if(hexcolor.length == 7){
					var [r, g, b] = [hexcolor.substring(1, 3), hexcolor.substring(3, 5), hexcolor.substring(5, 7)];
				}else{
					return hexcolor;
				}
				var [r, g, b] = [parseInt(r, 16), parseInt(g, 16), parseInt(b, 16)];
				return `rgba(${r}, ${g}, ${b}, ${alpha})`;
			}
		};
	},
	activated(){  // when component become active
		if (this.visible){
			this.react();
		}
	},
	watch: {
		// NOTE: There are several data variable that are watched dynamically
		// to avoid redrawing and recalculating the plot with recursive loops
		// See 'init' (calling 'turnWatchersOn')
		data: {
			immediate: true,
			handler(newval, oldval){
				this.visible = (typeof newval === 'object') && (Object.keys(newval).length);
				if (this.visible){ // see prop below
					this.init.call(this, newval);
				}
			}
		}
	},
	computed: {
		isGridCusomizable(){
			return Object.keys(this.gridlayouts).length>1;
		}
	},
	template: `<div v-show='visible' class='d-flex flex-row'>
		<div class="d-flex flex-column" style="flex: 1 1 auto">
			<div v-if="params.length" class='d-flex flex-row justify-content-around mb-3'>
				<template v-for='(param, index) in params'>
					<div v-if='param.label && param.value!==undefined && selectedgridlayout && !gridlayouts[selectedgridlayout].includes(param)'
						 class='d-flex flex-row align-items-baseline'
						 :class="index > 0 ? 'ms-2' : ''" style="flex: 1 1 auto">
						<span class='text-nowrap me-1'>{{ param.label }}</span>
						<select v-model="param.value" class='form-control' style="flex: 1 1 auto">
							<option v-for='value in param.values' :value="value">
								{{ value }}
							</option>
						</select>
					</div>
				</template>
			</div>
			<div class='position-relative' style="flex: 1 1 auto">
				<div :style='{display: drawingPlots ? "flex" : "none"}'
					 class='position-absolute start-0 top-0 end-0 bottom-0 flex-column align-items-center justify-content-center'
					 style='z-index:1001;background-color:rgba(0,0,0,0.0)'>
					<div class='p-2 shadow border rounded text-white d-flex flex-column align-items-center'
						 style="background-color:rgba(0,0,0,0.3)">
						<span v-html="waitbar.msg" class='border-0 bg-transparent' style="font-size:200%;"></span>
						<span class='border-0 bg-transparent'>(It might take a while, please wait)</span>
					</div>
				</div>
				<div class='position-absolute start-0 top-0 end-0 bottom-0' ref='rootDiv'
					 :id="'plot-div-' + new Date().getTime() + Math.random()"></div>
			</div>
		</div>
		<!-- RIGHT TOOLBAR (legend, buttons, controls) -->
		<div class='d-flex flex-column ps-4' v-show="legend.length || isGridCusomizable">
			<slot></slot> <!-- slot for custom buttons -->
			<div v-show='legend.length' class='mt-3 border p-2 bg-white px-1'
				 style='flex: 1 1 auto;overflow: auto'>
				<div>Legend</div>
				<div v-for="l in legend" class='d-flex flex-column'>
					<div class='d-flex flex-row align-items-baseline'  getLegendColor
						 :style="{color: getLegendColor(l[1])}">
						<label class='my-0 mt-2 text-nowrap' :class="{'checked': l[1].visible}"
							style='flex: 1 1 auto'>
							<input type='checkbox' v-model="l[1].visible"  getLegendColor
								   :style="{'accent-color': getLegendColor(l[1]) + ' !important'}"
								   @change="setTraceStyle(l[0], l[1])"> {{ l[0] }}
						</label>

						<div data-balloon-pos="left" data-balloon-length="small" class='ms-1'
						     aria-label='Style the plot traces (lines, bars, markers) of this legend group'>
							<i class="fa fa-chevron-down" style="cursor:pointer"
							   onclick='this.parentNode.parentNode.parentNode.querySelector("div._pso").classList.toggle("d-none"); this.classList.toggle("fa-chevron-up"); this.classList.toggle("fa-chevron-down")'></i>
						</div>
					</div>
					<div class='_pso d-flex flex-column d-none'>
						<textarea class='border' spellcheck="false"
								  style='margin:0px;padding:0px !important; height: 12rem;font-family:monospace; white-space: pre; overflow-wrap: normal; overflow-x: scroll; z-index:100; background-color: #f5f2f0;'
								  v-model="l[2]"/>
						<button type="button" class='mt-1 btn btn-sm' :disabled="!jsonParse(l[2])"
								@click="setTraceStyle(l[0], jsonParse(l[2]))"
								:style="{color: getLegendColor(l[1]), 'border-color': getLegendColor(l[1])}">Apply</button>
					</div>
				</div>
			</div>
			<div>
				<div class='mt-3 border p-2 bg-white'>
					<select @change="downloadTriggered" class='form-control'
							data-balloon-pos='left' data-balloon-length='medium'
							aria-label='Download the computed results in different formats. Notes: EPS images do not support color transparency, the result might not match what you see'>
						<option value="">Download as:</option>
						<option value="json">json</option>
						<option value="csv">text/csv</option>
						<option value="csv_eu">tex/csv (decimal comma)</option>
						<option value="png">png (visible plots only)</option>
						<option value="pdf">pdf (visible plots only)</option>
						<option value="eps">eps (visible plots only)</option>
						<option value="svg">svg (visible plots only)</option>
					</select>
				</div>
				<div v-show="isGridCusomizable" class='mt-3 border p-2 bg-white'>
					<div>Subplots layout</div>
					<select v-model='selectedgridlayout' class='form-control mt-1'>
						<option v-for='key in Object.keys(gridlayouts)' :value="key" v-html="key">
						</option>
					</select>
				</div>
				<div class='mt-3 d-flex flex-column border p-2 bg-white'>
					<div>Axis</div>
					<div v-for="type in ['x', 'y']" class='d-flex flex-row mt-1 text-nowrap align-items-baseline'>
						<span class='text-nowrap'>{{ type }}:</span>
						<label class='text-nowrap m-0 ms-2'
							   :class="{'checked': axisOptions[type].sameRange.value}"
							   :disabled="axisOptions[type].sameRange.disabled">
							<input type='checkbox' v-model='axisOptions[type].sameRange.value'
								   :disabled="axisOptions[type].sameRange.disabled"  class="me-1">
							<span>same range</span>
						</label>
						<label class='text-nowrap m-0 ms-2'
							   :class="{'checked': axisOptions[type].log.value}"
							   :disabled="axisOptions[type].log.disabled">
							<input type='checkbox' v-model='axisOptions[type].log.value'
								   :disabled="axisOptions[type].log.disabled" class="me-1">
							<span>log scale</span>
						</label>
						<label class='text-nowrap m-0 ms-2'
							   :class="{'checked': axisOptions[type].grid.value}"
							   :disabled="axisOptions[type].grid.disabled">
							<input type='checkbox' v-model='axisOptions[type].grid.value'
								   :disabled="axisOptions[type].grid.disabled" class="me-1">
							<span>grid</span>
						</label>
					</div>
				</div>
				<div class='mt-3 d-flex flex-column border p-2 bg-white'>
					<div> Mouse interactions</div>
					<div class='d-flex flex-row mt-1 align-items-baseline'>
						<span class='text-nowrap me-1'> on hover:</span>
						<select v-model="mouseMode.hovermode"
								class='form-control form-control-sm'>
							<option v-for='name in mouseMode.hovermodes' :value='name'>
								{{ mouseMode.hovermodeLabels[name] }}
							</option>
						</select>
					</div>
					<div class='d-flex flex-row mt-1 align-items-baseline'>
						<span class='text-nowrap me-1'> on drag:</span>
						<select v-model="mouseMode.dragmode"
								class='form-control form-control-sm'>
							<option v-for='name in mouseMode.dragmodes' :value='name'>
								{{ mouseMode.dragmodeLabels[name] }}
							</option>
						</select>
					</div>
				</div>
			</div>
		</div>
	</div>`,
	methods: {
		// methods to be overridden:
		getData(responseObject){
			/* Return from the given response object an Array of Objects representing
			the sub-plot to be visualized. Each sub-plot Object has the form:
			{traces: Array, params: Object, xaxis: Object, yaxis: Object}
			See residuals.js and trellis.js for a details docstring and implementation
			*/
		},
		displayGridLabels(axis, paramName, paramValues){
			/* Return true / false to specify if the given parameter should be displayed
			as grid parameter along the specified axis. In this default implementation,
			return true if `paramValues.length > 1`.
			Function arguments:
				`axis`: string, either 'x' or 'y'
				`paramName`: the string denoting the parameter name along the given axis
				`paramValues`: Array of the values of the parameter keyed by 'paramName'
			*/
			return paramValues.length > 1;
		},
		configureLayout(layout){
			/* Configure the `layout` Object to be passed to plotly (for details, see
			https://plotly.com/javascript/reference/layout/). Note that the `layout` keys
			`font.family` and `font.size` will be overwritten if present as they need to
			be equal to the HTML page in order to reliably calculate spaces and sizes.
			Function argument:
				`layout` Object copied from `this.defaultlayout` which can be modified
					here. Note that this function does not need to return any value
			*/
			return
		},
		// END OF OVERRIDABLE METHODS
		init(jsondict){
			// unwatch watchers, if any:
			this.watchOff();
			this.legend = {};
			// convert data:
			this.plots = this.getData(jsondict);
			// update selection, taking into account previously selected stuff:
			this.setupSelection();
			this.watchOn('params', function (newval, oldval) {
				this.newPlot();
			},{deep: true});
			this.watchOn('selectedgridlayout', function(newval, oldval){
				this.newPlot(); // which changes this.selectedParams which should call newPlot above
			});
			this.setupAxisOptions();
			this.createLegend();
			// now plot:
			this.newPlot(this.plots);
		},
		watchOff(...keys){
			// turns (dynamically created/destroyed) watchers off.
			// If keys (list of strings) is NOT provided, turns off and deletes all watchers
			var keys2delete = keys.length ? keys : Object.keys(this.watchers);
			for(var name of keys2delete){
				if (name in this.watchers){
					this.watchers[name]();
					delete this.watchers[name];
				}
			}
		},
		watchOn(key, callback, options){
			if (key in this.watchers){
				this.watchOff(key);
			}
			var watch = this.$watch(key, callback, options || {});
			this.watchers[key] = watch;
		},
		setupSelection(){
			// sets up selectable params, including those selectable as 'x grid' or 'y rid'.
			// called once from 'init'
			var paramvalues = new Map();
			this.plots.forEach(plot => {
				var plotParams = plot.params || {};
				for (var paramName of Object.keys(plotParams)){
					if(!paramvalues.has(paramName)){
						paramvalues.set(paramName, new Set());
					}
					paramvalues.get(paramName).add(plotParams[paramName]);
				}
			});
			var params = [];
			paramvalues.forEach((pvalues, pname) => {
				var values = Array.from(pvalues);
				if (values.every(v => v === null || typeof v === 'number')){  // https://stackoverflow.com/a/1063027
					values.sort((a, b) => b===null ? 1 : (a===null ? -1 : a - b));
				}else{
					values.sort();
				}
				params.push({
					values: values,
					label: pname,
					value: values[0],
					indexOf(plot, idx, plots){ return this.values.indexOf(plot.params[this.label]); }
				});
			});

			// dummy params that might be added below:
			var singleParam = {
				values: [''],  // Array of empty values (just to calculate plots width/height)
				indexOf(plot, idx, plots){
					return 0;
				}
			};

			// params with a single choosable value, and multi values:
			var multiValueParams = params.filter(p => p.values.length > 1);
			var singleValueParams = params.filter(p => p.values.length == 1);
			//var [paramNames, gridlabels2show] = [[], []];
			var gridlayouts = {};
			var selectedgridlayout = '';
			// DEFINITIONS:
			// SVP (single value param): a parameter that has the same value for all plots
			// MVP (multi value param): a parameter that has a unique value for each plot
			// (note that the case where two plots share the same param value should never happen,
			// but we cannot check and correct for it)
			if (multiValueParams.length == 0){
				if (plots.length == 1){
					// case 1 (see DEFINITIONS above): only N>=0 SVPs, and the plots count is one:
					// nothing complex to do: assure the param count is at least two,
					// take the first two and provide a single possible grid selection (based on those two parameters)
					// the grid selection will be hidden as nothing can be choosen
					// if we have a single SVP, and one plot, display the param name for that SVP
					// on the xgrid above the plot, as to emulate a title, if one wants to:
					params = [singleParam, singleParam];
					selectedgridlayout = '---';  // any value is irrelevant
					gridlayouts[selectedgridlayout] = [singleParam, singleParam];
				}else{
					// case 2 (see DEFINITIONS above): only N>=0 SVPs and the plots count is more than one:
					// this should not happen but nevertheless provide a 'stack horizontally' and
					// 'stack vertically' grid options, by the order specified by building a "fake"
					// multi-value-param which assigns an incremental index to each plot
					// assure SVPs are at least one:
					var multiParam = {
						values: plots.map(elm => ''), // Array of empty values (just to calculate plots width/height)
						indexOf(plot, idx, plots){
							return idx;
						}
					};
					params = [multiParam, singleParam];
					// set the two combinations of the SVP and the MVP as only choosable grid options
					// (providing a 'stack horizontally and 'stack vertically' generic names):
					gridlayouts['&harr; stack horizontally'] = [multiParam, singleParam];
					gridlayouts['&varr; stack vertically'] = [singleParam, multiParam];
					selectedgridlayout = '&varr; stack vertically';
				}
			}else{
				params = multiValueParams;
				// always provide a grid option selecting a single plot:
				gridlayouts['single plot'] = [singleParam, singleParam];
				if (multiValueParams.length == 1){
					// case 3 (see DEFINITIONS above): N>=0 SVPs and 1 MVP: basically same as
					// case 2 above but we do not need to create a fake multi-value param,
					// we use what we have providing the parameter name in the grid options
					// (horizontal vs vertical) with the parameter name
					// instead of generic 'stack horizontally' or 'stack vertically'
					selectedgridlayout = `&varr; ${multiValueParams[0].label}`;
					gridlayouts[selectedgridlayout] = [singleValueParam, multiValueParams[0]];
					gridlayouts[`&harr; ${multiValueParams[0].label}`] = [multiValueParams[0], singleValueParam];
				}else{
					// case 4 (see DEFINITIONS above): N>=0 SVPs and M>1 MVPs: build a choosable grid
					// with all combinations of the given MVPs times 2, as for each couple of P1, P2
					// we can display the grid as P1xP2 or P2xP1
					for (var prm1 of params){
						for (var prm2 of params){
							if (prm1 === prm2){
								continue;
							}
							var gridlayoutname = `&harr; ${prm1.label} vs. &varr; ${prm2.label}`;
							gridlayouts[gridlayoutname] = [prm1, prm2];
							if (!selectedgridlayout){ // take the first combination as selected one:
								selectedgridlayout = gridlayoutname;
							}
						}
					}
				}
			}
			// set defaults:
			this.gridlayouts = gridlayouts;
			this.selectedgridlayout = selectedgridlayout;
			this.params = params;
			this.newPlot();
		},
		setupAxisOptions(){
			// Initializes the values of this.axisOptions based on the plots we have. Axes
			// options are bound to checkbox controls on the side panel of the plot grid

			var keys = [
				'axisOptions.x.log.value',
				'axisOptions.y.log.value',
				'axisOptions.x.sameRange.value',
				'axisOptions.y.sameRange.value',
				'axisOptions.x.grid.value',
				'axisOptions.y.grid.value'
			];
			this.watchOff(...keys);

			// Reminder: this.plots is an Array of Objects of this type: {
			//	traces: [] // list of Plotly Objects representing traces
			//	xaxis: {}  // Plotly Object representing x axis
			//	yaxis: {}  // Plotly Object representing y axis
			// }
			this.plots.forEach(plot => {
				if (!plot.xaxis){ plot.xaxis={}; }
				if (!plot.yaxis){ plot.yaxis={}; }
			});

			var defaultPlotlyType = '-';
			var allAxisTypeUndefined = false; // will be set below

			this.axisOptions.x.log.disabled = false;
			// check for any plot P, P.xaxis.type ('-', 'linear', 'log'): enable the
			// x axis.log checkbox (on the right panel) if, for every P, P.axis.type is
			// missing or 'log'. In the latter case also force axis.log checkbox=true
			allAxisTypeUndefined = this.plots.every(p => [undefined, defaultPlotlyType].includes(p.xaxis.type));
			if (!allAxisTypeUndefined){
				var allAxisTypeAreLog = this.plots.every(p => p.xaxis.type === 'log');
				this.axisOptions.x.log.disabled = !allAxisTypeAreLog;
				if (allAxisTypeAreLog){
					this.axisOptions.x.log.value = true;
				}else{
					this.axisOptions.x.log.disabled = true;
				}
			}
			this.axisOptions.x.sameRange.disabled = this.plots.some(p => 'range' in p.xaxis);

			this.axisOptions.y.log.disabled = false;
			// check for any plot P, P.yaxis.type ('-', 'linear', 'log'): enable the
			// y axis.log checkbox (on the right panel) if, for every P, P.axis.type is
			// missing or 'log'. In the latter case also force axis.log checkbox=true
			allAxisTypeUndefined = this.plots.every(p => [undefined, defaultPlotlyType].includes(p.yaxis.type));
			if (!allAxisTypeUndefined){
				var allAxisTypeAreLog = this.plots.every(p => p.yaxis.type === 'log');
				if (allAxisTypeAreLog){
					this.axisOptions.y.log.value = true;
				}else{
					this.axisOptions.y.log.disabled = true;
				}
			}
			this.axisOptions.y.sameRange.disabled = this.plots.some(p => 'range' in p.yaxis);

			// restart watching:
			for (var key of keys){
				// watch each prop separately because with 'deep=true' react is called more than once ...
				this.watchOn(key, (newval, oldval) => { this.react(); });
			}
		},
		newPlot(){
			/**
			 * Filters the plots to display according to current parameters and grid choosen, and
			 * calls Plotly.newPlot on the plotly <div>
			 */
			var divElement = this.$refs.rootDiv;
			this.$nextTick(() => {
				var [hover, drag] = ['mouseMode.hovermode', 'mouseMode.dragmode'];
				this.watchOff(hover, drag);
				var [data, layout] = this.createPlotlyDataAndLayout(divElement);
				this.execute(function(){
					Plotly.newPlot(divElement, data, layout, this.defaultplotlyconfig);
					// now compute labels and ticks size:
					var newLayout = this.relayout(layout);
					Plotly.relayout(divElement, newLayout);
					this.watchOn(hover, function (newval, oldval) {
						this.setMouseModes(newval, undefined);  // hovermode, mousemode
					});
					this.watchOn(drag, function(newval, oldval){
						this.setMouseModes(undefined, newval);  // hovermode, mousemode
					});
				}, {delay: 300});  // this delay is increased because we might have the animation
				// of the form closing to be finished: overlapping wait-bar and form closing is
				// sometimes not nice
			});
		},
		react(){
			/**
			 * Same as this.newPlot above, and can be used in its place to create a plot,
			 * but when called again on the same <div> will update it far more efficiently
			 */
			var divElement = this.$refs.rootDiv;
			this.$nextTick(() => {
				this.execute(function(){
					var [data, layout] = this.createPlotlyDataAndLayout(divElement);
					Plotly.react(divElement, data, layout);
					var newLayout = this.relayout(layout);
					Plotly.relayout(divElement, newLayout);
				});
			});
		},
		execute(callback, options){
			// Executes asynchronously the given callback (which can safely use `this`
			// in its code to point to this Vue component) showing a wait bar meanwhile.
			// 'options' is an Object with two optional properties:
			// options.msg (the wait bar message)
			// and delay (the execution delay)
			var delay = (options || {}).delay || 200;
			this.waitbar.msg = (options || {}).msg || this.waitbar.DRAWING;
			this.drawingPlots=true;
			setTimeout(() => {
				callback.call(this);
				this.drawingPlots=false;
			}, delay);
		},
		createPlotlyDataAndLayout(divElement){
			var plots = this.plots;
			var [gridxparam, gridyparam] = this.gridlayouts[this.selectedgridlayout];
			// filter plots according to the value of the parameter which are not displayed as grid param:
			for (var param of this.params){
				if (param === gridxparam || param === gridyparam || param.value === undefined){
					continue;
				}
				plots = plots.filter(plot => plot.params[param.label] == param.value);
			}
			// console.log('creating plots');
			this.setupPlotAxis(plots);

			// now build an array the same length as plots with each element the grids position [index_x, index_y]
			var plotsGridIndices = plots.map((plot, idx, plots) =>
				[gridxparam.indexOf(plot, idx, plots), gridyparam.indexOf(plot, idx, plots)]);
			var layout = Object.assign({}, this.defaultlayout);
			this.configureLayout(layout);
			// synchronize hovermode and hovermode
			// between the layout and this.mouseMode:
			['hovermode', 'dragmode'].forEach(elm => {
				if (elm in layout){
					this.mouseMode[elm] = layout[elm];
				}else{
					layout[elm] = this.mouseMode[elm];
				}
			});
			// set font (if not present):
			if (!('font' in layout)){
				layout.font = {};
			}
			// setup font as the body font, and size as set in the object properties, since the font size
			// is used to calculate spaces it cannot be changed in configureLayout:
			layout.font.family = window.getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('font-family');
			layout.font.size = this.plotfontsize;
			var data = [];
			// compute rows, cols, and margins for paramsGrid labels:
			var colwidth = 1.0;
			var rowheight = 1.0;
			var marginleft = 0;
			var marginbottom = 0;

			if (gridxparam.label || gridyparam.label){
				var [width, height] = this.getElmSize(this.$refs.rootDiv);
				var margin = this.paramsGridMargin;
				if (gridxparam.label){
					marginbottom = margin / height;
				}
				if (gridyparam.label){
					marginleft = margin / width;
				}
			}
			var cols = gridxparam.values.length;
			if (cols > 1 || marginleft){
				colwidth = (1-marginleft) / cols;
			}
			var rows = gridyparam.values.length;
			if (rows > 1 || marginbottom){
				rowheight = (1-marginbottom) / rows;
			}

			var legendgroups = new Set();
			for (var i = 0; i < plots.length; i++){
				var [plot, [gridxindex, gridyindex]] = [plots[i], plotsGridIndices[i]];
				// compute domains (assure the second domain element is 1 and not, e.g., 0.9999):
				var xdomain = [marginleft + gridxindex*colwidth, 1+gridxindex == cols? 1 : marginleft+(1+gridxindex)*colwidth];
				var ydomain = [marginbottom + gridyindex*rowheight, 1+gridyindex == rows ? 1 : marginbottom+(1+gridyindex)*rowheight];

				var axisIndex = 1 + gridyindex * cols + gridxindex;
				var xaxis = { domain: xdomain, anchor: `y${axisIndex}`, showgrid: this.axisOptions.x.grid.value };
				var yaxis = { domain: ydomain, anchor: `x${axisIndex}`, showgrid: this.axisOptions.y.grid.value };

				// merge plot xaxis defined in getData with this.defaultxaxis, and then with xaxis.
				// Priority in case of conflicts goes from right (xaxis) to left (this.defaultxaxis)
				layout[`xaxis${axisIndex}`] = xaxis = Object.assign({}, this.defaultxaxis, plot.xaxis, xaxis);
				// merge plot yaxis defined in getData with this.defaultyaxis, and then with yaxis.
				// Priority in case of conflicts goes from right (yaxis) to left (this.defaultyaxis)
				layout[`yaxis${axisIndex}`] = yaxis = Object.assign({}, this.defaultyaxis, plot.yaxis, yaxis);
				// Now edit the traces in plot.traces. Each trace is basically an Object of the
				// form T = {x: ..., y: ...}. These trace properties need to be modified EACH TIME:
				// 1. T.xaxis and T.yaxis (i.e., basically on which subplot the trace has to be displayed), and
				// 2. T.showlegend. The latter is set to true for the FIRST trace found for a given legendgroup
				// in order to avoid duplicated legend names. (T.legendgroup = T.name is the GSIM name, and it allows
				// toggling visibility simultaneously on all subplots for every trace with the same legendgroup, when
				// clicking on the legend's GSIM name)
				plot.traces.forEach(function(element){
					element.xaxis = `x${axisIndex}`;
					element.yaxis = `y${axisIndex}`;
					if ('legendgroup' in element){
						element.showlegend = !legendgroups.has(element.legendgroup);
						legendgroups.add(element.legendgroup);
					}
					data.push(element);
				});
				/* set standoff property (distance label text and axis) */
				if (xaxis.title){
					if(typeof xaxis.title !== 'object'){
						xaxis.title = {text: `${xaxis.title}`};
					}
					xaxis.title.standoff = 10; // space between label and axis
				}
				if (yaxis.title){
					if(typeof yaxis.title !== 'object'){
						yaxis.title = {text: `${yaxis.title}`};
					}
					yaxis.title.standoff = 5; // space between label and axis
				}
			}
			return [data, layout];
		},
		setupPlotAxis(plots){
			// sets up the plotly axis data on the plots to be plotted, according to
			// the current axis setting and the plot data

			// set axis type according to the selcted checkbox:
			var defaultPlotlyAxisType = '-';
			var isXAxisLog = (!this.axisOptions.x.log.disabled) && this.axisOptions.x.log.value;
			var isYAxisLog = (!this.axisOptions.y.log.disabled) && this.axisOptions.y.log.value;
			plots.forEach(plot => {
				plot['xaxis'].type = isXAxisLog ? 'log' : defaultPlotlyAxisType;
				plot['yaxis'].type = isYAxisLog ? 'log' : defaultPlotlyAxisType;
			});
			// setup ranges:
			var [sign, log10] = [Math.sign, Math.log10];
			for (var key of ['x', 'y']){
				var axisOpt = this.axisOptions[key];  // the key for this,.axis ('x' or 'y')
				var axisKey = key + 'axis';  //  the key for each plot axis: 'xaxis' or 'yaxis'
				// set same Range disabled, preventing the user to perform useless click:
				// Note that this includes the case of only one plot:
				if (!axisOpt.sameRange.value || axisOpt.sameRange.disabled){
					plots.forEach(plot => {
						delete plot[axisKey].range;
					});
					continue;
				}
				// here deal with the case we have 'sameRange' clicked (and enabled):
				var range = [NaN, NaN];
				plots.forEach(plot => {
					var [rangex, rangey] = this.getPlotBounds(plot.traces || []);  // note that hidden traces count
					range = this.nanrange(...range, ...(key == 'x' ? rangex : rangey));
				});

				// add margins for better visualization:
				var margin = Math.abs(range[1] - range[0]) / 50;
				// be careful with negative logarithmic values:
				var isAxisLog = key === 'x' ? isXAxisLog : isYAxisLog;
				if (!isAxisLog || (range[0] > margin && range[1] > 0)){
					range[0] -= margin;
					range[1] += margin;
				}

				// set computed ranges to all plot axis:
				if (!isNaN(range[0]) && !isNaN(range[1])){
					plots.forEach(plot => {
						// plotly wants range converted to log if axis type is 'log':
						plot[axisKey].range = plot[axisKey].type === 'log' ? [log10(range[0]), log10(range[1])] : range;
					});
				}
			}
		},
		relayout(layout){
			var [gridxparam, gridyparam] = this.gridlayouts[this.selectedgridlayout];
			var xdomains = gridxparam.label ? new Array(gridxparam.values.length) : [];
			var ydomains = gridyparam.label ? new Array(gridyparam.values.length) : [];
			var margin = this.getPlotsMaxMargin();
			var newLayout = {};
			for (var key of Object.keys(layout)){
				if (!key.startsWith('xaxis') && !key.startsWith('yaxis')){
					continue;
				}
				var domain = layout[key].domain;
				if (!domain){
					continue;
				}
				var plotIndex = key.substring(5);
				plotIndex = plotIndex ? parseInt(plotIndex)-1 : 0;

				if (key.startsWith('x')){
					newLayout[`${key}.domain`] = [domain[0]+margin.left, domain[1]-margin.right];
					if (gridxparam.label){
						if (plotIndex < gridxparam.values.length){
							xdomains[plotIndex] = domain;
						}
					}
				}else{
					newLayout[`${key}.domain`] = [domain[0]+margin.bottom, domain[1]-margin.top];
					if (gridyparam.label){
						var cols = gridxparam.values.length;
						if (plotIndex % cols == 0){
							ydomains[parseInt(plotIndex / cols)] = domain;
						}
					}
				}
			}

			var defAnnotation = {
				xref: 'paper',
				yref: 'paper',
				showarrow: false,
				font: {size: 1.2*this.plotfontsize}
			};

			newLayout.annotations = Array.from(this.defaultlayout.annotations);
			var [width, height] = this.getElmSize(this.$refs.rootDiv);
			if (gridxparam.label){
				for (var i=0; i < gridxparam.values.length; i++){
					var domain = xdomains[i];
					var w = width - (gridyparam.label ? this.paramsGridMargin : 0);
					// avoid label overlapping by removing 3 pixel from left and right:
					w-= 6 * gridxparam.values.length;
				 	newLayout.annotations.push(Object.assign({}, defAnnotation, {
						x: (domain[1] + domain[0])/2,
						y: 0,
						width: w/gridxparam.values.length,
						height: 2*this.paramsGridMargin/3,
						xanchor: 'center', /* DO NOT CHANGE THIS */
						yanchor: 'bottom',
						text: `${gridxparam.label}: ${gridxparam.values[i]}`,
						bgcolor: 'rgba(0,102,133,0.1)',
						borderwidth: 1,
						bordercolor: 'rgba(0,102,133,0.4)',
					}));
				}
			}
			if (gridyparam.label){
				for (var i=0; i < gridyparam.values.length; i++){
					var domain = ydomains[i];
					var w = height - (gridxparam.label ? this.paramsGridMargin : 0);
					// avoid label overlapping by removing 3 pixel from top and bottom:
					w-= 6 * gridyparam.values.length;
				 	newLayout.annotations.push(Object.assign({}, defAnnotation, {
						x: 0,
						y: (domain[1] + domain[0])/2,
						width: w/gridyparam.values.length,
						height: 2*this.paramsGridMargin/3,
						xanchor: 'left',
						yanchor: 'middle', /* DO NOT CHANGE THIS */
						text: `${gridyparam.label}: ${gridyparam.values[i]}`,
						textangle: '-90',
						bgcolor: 'rgba(0,102,133,0.1)',
						borderwidth: 1,
						bordercolor: 'rgba(0,102,133,0.4)',
					}));
				}
			}
			return newLayout;
		},
		getPlotsMaxMargin(){
			// Return an object representing the max margins of all plots, where
			// each plot margin is computed subtracting the outer axes rectangle
			// (plot area + axis ticks and ticklabels area) and the inner one
			var margin = { top: 20, bottom: 0, right: 10, left: 0 };
			var [min, max, abs] = [Math.min, Math.max, Math.abs];
			var plotDiv = this.$refs.rootDiv;
			var certesianLayer = plotDiv.querySelector('g.cartesianlayer');
			var infoLayer = plotDiv.querySelector('g.infolayer');
			for (var elm of certesianLayer.querySelectorAll('g[class^=subplot]')){
				// get plot index from classes of type 'xy' 'x2y2' and so on:
				var xindex = '';
				var yindex = '';
				var re = /^(x\d*)(y\d*)$/g;
				for (var cls of elm.classList){
					var matches = re.exec(cls);
					if (matches){
						xindex = matches[1];
						yindex = matches[2];
						break;
					}
				}
				if (!xindex || !yindex){
					continue;
				}
				var innerPlotRect = elm.querySelector(`path.ylines-above.crisp`) || elm.querySelector(`path.xlines-above.crisp`);
				if(!innerPlotRect){
					continue;
				}
				innerPlotRect = innerPlotRect.getBBox();

				// try to find the xlabel, otherwise get the xticks+xticklabels:
				var xlabel = infoLayer.querySelector(`g[class=g-${xindex}title]`) || elm.querySelector('g.xaxislayer-above');
				if (xlabel){
					var xElm = xlabel.getBBox();
					margin.bottom = max(margin.bottom, xElm.y + xElm.height - innerPlotRect.y - innerPlotRect.height);
				}
				// try to find the ylabel, otherwise get the yticks+yticklabels:
				var ylabel =  infoLayer.querySelector(`g[class=g-${yindex}title]`)  || elm.querySelector('g.yaxislayer-above');
				if (ylabel){
					var yElm = ylabel.getBBox();
					// margin.top = max(margin.top, axesRect.y - yElm.y);
					margin.left = max(margin.left, innerPlotRect.x - yElm.x);
				}
			}

			var [width, height] = this.getElmSize(this.$refs.rootDiv);
			margin.left /= width;
			margin.right /= width;
			margin.top /= height;
			margin.bottom /= height;

			return margin;
		},
		getElmSize(domElement){
			// returns the Array [width, height] of the given dom element size
			return [domElement.offsetWidth, domElement.offsetHeight];
		},
		createLegend(){
			this.legend = [];
			var legend = this.legend;
			var legendgroups = new Set();
			this.plots.forEach(function(plot, i){
				plot.traces.forEach(function(trace){
					var legendgroup = trace.legendgroup;
					if (legendgroup && !legendgroups.has(legendgroup)){
						legendgroups.add(legendgroup);
						var legenddata = {visible: ('visible' in trace) ? !!trace.visible : true};
						for (var key of ['line', 'marker']){
							if (key in trace){
								legenddata[key] = Object.assign({}, trace[key]);
							}
						}
						legend.push([legendgroup, legenddata, JSON.stringify(legenddata, null, '  ')]);
					}
				});
			});
		},
		getLegendColor(legenddata){
			if (legenddata) {
				var marker = legenddata.marker;
				if (marker && marker.line && marker.line.color){
					return marker.line.color;
				}
				if (legenddata.line && legenddata.line.color){
					return legenddata.line.color;
				}
				if (marker && marker.color){
					return marker.color;
				}
			}
			return '#000000';
		},
		setTraceStyle(legendgroup, legenddata){
			if (!legenddata){ return; }
			for (var legend of this.legend){
				if(legend[0] === legendgroup){
					legend[1] = legenddata;
					legend[2] = JSON.stringify(legenddata, null, "  ")
				}
			}
			var indices = [];
			var plotlydata = this.getPlotlyDataAndLayout()[0];
			plotlydata.forEach(function(data, i){
				if (data.legendgroup === legendgroup){
					indices.push(i);
				}
			});
			if(indices.length){
				this.execute(function(){
					Plotly.restyle(this.$refs.rootDiv, legenddata, indices);
				});
			}
		},
		jsonParse(jsonString){
			try{
				return JSON.parse(jsonString);
			}catch(error){
				return null;
			}
		},
		getPlotlyDataAndLayout(){
			/**
			 * returns [data, layout] (Array an Object),
			 * i.e. the Plotly data and layout used to draw the plot
			 */
			var elm = this.$refs.rootDiv;
			return elm ? [elm.data || [], elm.layout || {}] : [[], {}];
		},
		getPlotBounds(traces){
			// gets the bounds min max of all the traces of 'plot', which must be an
			// element of this.plots. Returns [rangeX, rangeY] where each range has two numeric elements
			// pmin, max] which might be NaN
			var [rangeX, rangeY] = [[NaN, NaN], [NaN, NaN]];
			for (var trace of traces){
				if (trace.x){
					var rangex = this.nanrange(...trace.x);
					rangeX = this.nanrange(...rangeX, ...rangex);
				}
				if (trace.y){
					var rangey = this.nanrange(...trace.y);
					rangeY = this.nanrange(...rangeY, ...rangey);
				}
			}
			return [rangeX, rangeY];
		},
		nanrange(...values){
			values = values.filter(elm => typeof elm === 'number' && !isNaN(elm));
			var m = Math;
			return values.length? [m.min(...values), m.max(...values)] : [NaN, NaN];
		},
		setMouseModes(hovermode, dragmode){
			var [data, layout] = this.getPlotlyDataAndLayout();
			var relayout = false;
			if (this.mouseMode.hovermodes.includes(hovermode)){
				layout.hovermode = hovermode;
				relayout = true;
			}
			if (this.mouseMode.dragmodes.includes(dragmode)){
				layout.dragmode = dragmode;
				relayout = true;
			}
			if (relayout){
				this.execute(function(){
					Plotly.relayout(this.$refs.rootDiv, layout);
				}, {msg: this.waitbar.UPDATING});
			}
		},
		downloadTriggered(event){
			var selectElement = event.target;
			if (selectElement.selectedIndex == 0){
				return;
			}
			var format = selectElement.value;
			var url = this.downloadUrl + '.' + format;
			var data = this.data;
			if (format == 'json'){
				var filename =  url.split('/').pop();
				this.saveAsJSON(data, filename);
			} else if (format.startsWith('csv')){
				this.download(url, data);
			}else{
				// image format:
				var [data, layout] = this.getPlotlyDataAndLayout();
				var parent = this.$refs.rootDiv.parentNode.parentNode.parentNode;
				var [width, height] = this.getElmSize(parent);
				data = data.filter(elm => elm.visible || !('visible' in elm));
				postData = {data:data, layout:layout, width:width, height:height};
				this.download(url, postData);
			}
			selectElement.selectedIndex = 0;
		}
	}
};