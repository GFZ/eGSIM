/**
 * Base "class" for any component showing plots as a result of a response Object sent from the server
 */
var _PLOT_DIV = Vue.component('plotdiv', {
    props: {
        data: {type: Object, default: () => { return{} }},
        // this is used to calculate plot areas and set it in the default layout (use computed body font-size. Note that parseInt('16px')=16):
        plotfontsize: {type: Number, default: parseInt(window.getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('font-size'))},
        filename: {type: String, default: 'egsim_plot'}
    },
    data: function(){
        // unique id based on the component name (note that if this is called by a subclass, then this.$options.name is the subclass name)
        var id = this.$options.name + new Date().getTime().toString();
        return {
            visible: !Vue.isEmpty(this.data),
            resizeListener: null,  //callback to be passed to window.addEventListener('resize', ...). See addResizeListener and removeResizeListener
            watchers: {},  // dynamically created watchers (see init)
            paramnames2showongrid: new Set(),  // parameter names to show on the grid
            plotdivid: Date.now().toString(),
            // NOTE: do not prefix data variable with underscore: https://vuejs.org/v2/api/#data
            legend: {}, // an Object of names -> html color (string) values
            plots: [], //array of [{traces, params}], where traces is an Array of Objects representing the plot traces, and params is an Object
            params: {},  // a dict of property names mapped to an array of possible (string) values
            gridlayouts: {}, // dict of subplots layout (string) mapped to a two element array (xgrid param name, y grid param name)
            selectedgridlayout: '',  // string denoting the selected key of gridLayouts
            // selectedParams below is a dict of property names mapped to a scalar denoting the selected value
            // it is the keys of this.params without the values of this.gridlayouts[this.selectedgridlayout]
            // and each param will be displayed on one single-value-choosable combobox
            selectedParams: {},
            // defaultlayout. Note that defaultlayout.annotations, if specified here, will be copied and then
            // xaxis and grid labels will be copied to the new copied Array before passing it to Plotly as layout argument
            defaultlayout: {
                autosize: true,
                paper_bgcolor: 'rgba(0,0,0,0)', //window.getComputedStyle(document.getElementsByTagName('body')[0]).getPropertyValue('background-color'),
                // font: {family: "Encode Sans Condensed, sans-serif", size: 12}, // this will be overridden
                showlegend: false,
                legend: { bgcolor: 'rgba(0,0,0,0)'},
                margin: {r: 0, b: 0, t: 0, l:0, pad:0},
                annotations: []
            },
            mouseMode: { // https://plot.ly/python/reference/#layout-hovermode
	            hovermodes: ["closest", "x", "y", false],
	            // the labels of hovermodes to be displayed. Copied from plotly modebar after visual test:
	            hovermodeLabels: {closest: 'show closest point', x: 'show x', y: 'show y', false: 'do nothing'},
       			dragmodes: ["zoom", "pan"],  // "select", "lasso" are useless. false does not seem to work (it's zoom)
 				dragmodeLabels: {zoom: 'zoom', pan: 'pan'},
       			hovermode: 'closest',  // will set this value to the Plotly layout before plotting, if not explicitly set
 				dragmode: 'zoom'  // will set this value to the Plotly layout before plotting, if not explicitly set
            },
            // axis options:
            axis: {
            	// reminder: x.log and y.log determine the type of axis. Plotly has xaxis.type that can be:
            	// ['-', 'linear', 'log', ... other values ], we will set here only 'normal' (log checkbox unselected)
            	// or log (log checkbox selected)
        		x: {
            		log: {disabled: false, value: undefined},  // undefined: infer from plots, null: disable the option, otherwise boolean
            		sameRange: {disabled: false, value: undefined}, // same as above
            	},
            	y: {
            		log: {disabled: false, value: undefined}, // see above
            		sameRange: {disabled: false, value: undefined} // see above
            	}
            },
            // the plotly config for plots. See
            // https://community.plot.ly/t/remove-options-from-the-hover-toolbar/130/14
            defaultplotlyconfig: {
                responsive: true,
                modeBarButtonsToRemove: ['sendDataToCloud', 'toImage'],
                displaylogo: false
            },
            defaultxaxis: {mirror: true, zeroline: false, linewidth: 1},  // domain and anchor properties will be overridden
            defaultyaxis: {mirror: true, zeroline: false, linewidth: 1},  // domain and anchor properties will be overridden
            colorMap: Vue.colorMap(),  // defined in vueutil.js,
            downloadopts: { // most of the image options are not user defined for the moment (except format):
                show: false,
                width: null,
                height: null,
                filename: this.filename,
                format: "",
                formats: ['', 'png', 'jpeg', 'svg', 'json (all plots data)'],
                // scale 5 increases the image size, thus increasing the resolution
                // Note that you cannot provide any number (e.g. scale 10 raises Exception)
                scale: 5
            }
        }
    },
    watch: {
        // NOTE: There are several data variable that are watched dynamically
        // to avoid redrawing and recalculating the plot with recursive loops
        // See 'init' (calling 'turnWatchersOn')
        data: {
            immediate: true,
            handler(newval, oldval){
                this.visible = !Vue.isEmpty(this.data);
                if (this.visible){ // see prop below
                    this.init.call(this, this.data);
                    this.addResizeListener(); // start redrawing plots on resize, see bottom of the file
                    // should be better added inside a this.$nextTick(() => { this.addResizeListener(); } ?
                }else{
                	this.removeResizeListener();  // stop redrawing plots on resize, see bottom of the file
                }
            }
        },
        'downloadopts.format': function (newVal, oldVal){
            // to work with changes in downloadopts.format: https://stackoverflow.com/a/46331968
            // we do not attach onchange on the <select> tag because of this: https://github.com/vuejs/vue/issues/293
            if(newVal){
                this.download();
            }
        },
    },
    activated: function(){  // when component become active
    	if (this.visible){
    		this.addResizeListener(true); // start redrawing plots on resize, see bottom of the file
    		// should be better added inside a this.$nextTick(() => { this.addResizeListener(); } ?
    	}
    },
    deactivated: function(){   // when component is deactivated
    	this.removeResizeListener(); // stop redrawing plots on resize, see bottom of the file
    },
    computed: {  // https://stackoverflow.com/a/47044150
        legendNames: function(){
            return Object.keys(this.legend);
        },
        isGridCusomizable: function(){
            return Object.keys(this.gridlayouts).length>1;
        }
    },
    template: `<div v-show='visible' class='d-flex flex-row'>
        
        <div class='flexible d-flex flex-column m-2'>
            <div v-if="Object.keys(selectedParams).length" class='d-flex flex-row justify-content-around mb-1'>
                <div v-for='(values, key) in selectedParams' class='d-flex flex-row flexible align-items-baseline ml-2'>
                    <span class='text-nowrap mr-1'>{{ key }}</span>
                    <select class='flexible form-control' v-model="selectedParams[key]">
                        <option v-for='value in params[key]' :value="value">
                            {{ value }}
                        </option>
                    </select>
                </div>
            </div>
            <div class='flexible position-relative'>
                <div class='position-absolute pos-0' :id="plotdivid"></div>
            </div>
        </div>
    
        <div class='d-flex flex-column my-2 mr-2 pl-3'
            v-show="legendNames.length || isGridCusomizable">

            <slot></slot> <!-- slot for custom buttons -->
            
            <div v-show='legendNames.length' class='flexible mt-3 border p-2 bg-white'>
                <div>Legend</div>
                <div v-for="key in legendNames">
                    <label v-bind:style="{color: legend[key]}" class='my-0 mt-2'>
                        <input type='checkbox' v-bind:checked="isTraceVisible(key)" v-on:click="toggleTraceVisibility(key)">
                        {{ key }}
                    </label>
                </div>
            </div>

			<div>
				<select v-model='downloadopts.format' class='form-control mt-3'>
	                <option v-for='(format, index) in downloadopts.formats' :value='format'
	                    :disabled='format == downloadopts.formats[0]'>
	                    {{ index ? format : 'download as:' }}
	                </option>
	            </select>
	
				<div v-show="isGridCusomizable" class='mt-3 border p-2 bg-white'>
	                <div>Subplots layout</div>
	                <select class='form-control mt-1' v-model='selectedgridlayout'>
	                    <option v-for='key in Object.keys(gridlayouts)' v-bind:value="key" v-html="key">
	                    </option>
	                </select>
	
	            </div>
	
				<div class='mt-3 d-flex flex-column border p-2 bg-white'>
					<div>Axis</div>
					<div v-for="type in ['x', 'y']" class='d-flex flex-row mt-1 text-nowrap align-items-baseline'>
						<span class='text-nowrap'>{{ type }}:</span>
	            		<label
	            			class='text-nowrap m-0 ml-2'
	            			:disabled="axis[type].sameRange.disabled"
	            		>
		            		<input
		            			type='checkbox'
		            			v-model='axis[type].sameRange.value'
		            			:disabled="axis[type].sameRange.disabled"
		            		>	
		            		<span class='ml-1'>same range</span>
		            	</label>
	            		<label
	            			class='text-nowrap m-0 ml-2'
	            			:disabled="axis[type].log.disabled"
	            		>
		            		<input
		            			type='checkbox'
		            			v-model='axis[type].log.value'
		            			:disabled="axis[type].log.disabled"
		            		>	
		            		<span class='ml-1'>log scale</span>
		            	</label>
	            	</div>
	            </div>
	
				<div class='mt-3 d-flex flex-column border p-2 bg-white'>
					<div> Mouse interactions</div>
					<div class='d-flex flex-row mt-1 align-items-baseline'>
						<span class='text-nowrap mr-1'> on hover:</span>
	            		<select v-model="mouseMode.hovermode" class='form-control form-control-sm'>
	            			<option v-for='name in mouseMode.hovermodes' :value='name'>{{ mouseMode.hovermodeLabels[name] }}</option>
	            		</select>
	            	</div>
	            	<div class='d-flex flex-row mt-1 align-items-baseline'>
						<span class='text-nowrap mr-1'> on drag:</span>
	            		<select v-model="mouseMode.dragmode" class='form-control form-control-sm'>
	            			<option v-for='name in mouseMode.dragmodes' :value='name'>{{ mouseMode.dragmodeLabels[name] }}</option>
	            		</select>
	            	</div>
	            </div>
 			</div>
        </div>
     
    </div>`,
    created: function() { // https://vuejs.org/v2/api/#created
        // no-op
    },
    methods: {
        // methods to be overridden:
        getData: function(responseObject){
            // this method needs to be implemented in order to initialize the response object into
            // an Array of sub-plots, and return that Array: each Array element (or "sub-plot")
            // need to be a js Object of the form:
            // {traces: Array, params: Object, xaxis: Object, yaxis: Object} where:
            //
            // traces:
            // an Array of valid representable plotly objects e.g. {x: Array, y: Array, name: string}
            // (providing a `name` key makes the name showing when hovering over the trace with
            // the mouse. For other keys, refer to plotly doc for details). The object keys
            // `xaxis`, 'yaxis', 'showlegend' needs not to be specified as they will be overridden.
            // NOTE: To add a unique color mapped to a trace id (e.g. the trace name) and
            // setup the legendgroup and automatically map the trace to a legend item toggling
            // the trace visibility, use `this.addLegend(trace, key)`, e.g.:
            //
            //   var trace = {x: Array, y: Array, name: 'mytrace'}
            //   var color = this.addLegend(trace, trace.name)
            //   trace.line = {color: color}  // set the trace color to the legend assigned color
            //
            // `addLegend(trace, K)` maps the returned color to the key K provided;
            // subsequent calls to this.addLegend(..., K) return the same color.
            // The returned color is a color assigned to K by cycling through an internal color
            // array (copied from plotly). If you want to specify a default
            // color for non-mapped keys avoiding the default assignement, call:
            // `addLegend(trace, key, color)` with an optional color string in the form '#XXXXXX'.
            // `addLegend` also sets trace.legendgroup=K.
            //
            // params:
            // an Object of selectable params (string) mapped to the plot specific values
            // (e.g. {magnitude: 5, 'xlabel: 'PGA'}).
            //
            // xaxis:
            // a dict of x axis properties. Example: {title: 'plottitle', type: 'log'}
            // The properties 'domain' and 'anchor' needs not to be specified as they will be overridden. 
            // The Object returned here will be merged with the properties defined this.defaultxaxis
            // (in case of conflicts, the properties of this.defaultxaxis will be overridden)
            //
            // yaxis:
            // a dict of y axis properties. Example: {title: 'plottitle', type: 'log'}
            // The properties 'domain' and 'anchor' needs not to be specified as they will be overridden. 
            // The Object returned here will be merged with the properties defined this.defaultyaxis
            // (in case of conflicts, the properties of this.defaultyaxis will be overridden)
        },
        displayGridLabels: function(axis, paramName, paramValues){
            // this optional method can be implemented to hide the labels of 'paramName' when it is
            // set as grid parameter along the specified axis ('x' or 'y'). If not implemented,
            // by default this method returns true if `paramValues.length > 1` because display the
            // only parameter value on the grid would be redundant and space consuming
            //
            // params:
            // 
            // axis: string, either 'x' or 'y'
            // paramName: the string denoting the parameter name along the given axis
            // paramValues: Array of the values of the parameter keyed by 'paramName'
            return paramValues.length > 1;
        },
        configureLayout: function(layout){
            // this optional method can be implemented to remove or add properties to the plotly `layout`
            // Object, which is passed as shallow copy of `this.defaultlayout` (defined in `plotsdiv`).
            // If not implemented, this method by default does nothing and returns.
            // Note that the layout font size and family will be set according to `this.plotfontsize`
            // and the <body> font family, because the font size is used to calculate plots area
            // and cannot be changed. 
            return
        },
        // END OF OVERRIDABLE METHODS
        addLegend: function(trace, key, defaultColor){
            // defaultColor is optional. If given (not undefined), it is in the form '#XXXXXX'
            var color;
            var colorMap = this.colorMap;
            if (defaultColor !== undefined && !colorMap.has(key)){
                colorMap.set(key, defaultColor);
                color = defaultColor;
            }else{
                color = colorMap.get(key);  // sets also the key if not existing
            }
            this.legend[key] = color;
            trace.legendgroup = key;
            return color;
        },
        init: function(jsondict){
            // unwatch watchers, if any:
            this.watchOff();
            this.legend = {};
            // convert data:
            this.plots = this.getData(jsondict);
            // update selection, taking into account previously selected stuff:
            this.setupSelection(); // which sets this.selectedgridlayout which calls this.gridLayoutChanged (see watch above)
            // which in turn sets this.selectedParams which eventually calls this.newPlot (see watch above)
            this.watchOn('selectedParams', function (newval, oldval) {
                this.newPlot();
            },{deep: true});
            this.watchOn('selectedgridlayout', function(newval, oldval){
                this.gridLayoutChanged(); // which changes this.selectedParams which should call newPlot above
            });
            
            this.setupAxisDefaults();
            // now plot:
            this.newPlot();
        },
        watchOff: function(...keys){
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
        watchOn: function(key, callback, options){
        	if (key in this.watchers){
        		this.watchOff(key);
        	}
        	var watch = this.$watch(key, callback, options || {});
        	this.watchers[key] = watch;
        },
        setupSelection: function(){
            // sets up selectable params, including those choosable as 'x grid' or 'y rid'.
            // called once from 'init'
            var plots = this.plots;
            var allparams = {};
            plots.forEach(plotElement => {
               var plotParams = plotElement.params;
               for (var key of Object.keys(plotParams)){
                   var paramValue = plotParams[key];
                   if(!(key in allparams)){
                       allparams[key] = new Set();
                   }
                   allparams[key].add(paramValue);
               }
            });
            
            // adds a new parameter name to each plot and to all plot
            // argument multiValue is boolean (false: single value)
            var addParam = function(multiValue){
                var newParamName = '';
                while (newParamName in allparams){
                    newParamName += ' ';
                }
                allparams[newParamName] = new Set();
                for (var i=0; i < plots.length; i++){
                    var val = multiValue ? i : 0;
                    plots[i].params[newParamName] = val;
                    allparams[newParamName].add(val);
                }
                return newParamName;
            }
            // param names with a single choosable value, and multi values:
            var multiValueParamNames = Object.keys(allparams).filter(key => allparams[key].size > 1);
            var singleValueParamNames = Object.keys(allparams).filter(key => allparams[key].size == 1);

            var [paramNames, gridlabels2show] = [[], []];
            
            var gridlayouts = {};
            var selectedgridlayout = '';
            
            // set the param names to be choosen.
            // DEFINITIONS:
            // SVP (single value param): a parameter that has the same value for all plots
            // MVP (multi value param): a parameter that has a unique value for each plot
            // (note that the case where two plots share the same param value should never happen,
            // but we cannot check and correct for it)
            
            if (multiValueParamNames.length == 0){
                if (plots.length == 1){
                    
                    // case 1 (see DEFINITIONS above): only N>=0 SVPs, and the plots count is one:
                    // nothing complex to do: assure the param count is at least two,
                    // take the first two and provide a single possible grid selection (based on those two parameters)
                    // the grid selection will be hidden as nothing can be choosen

                    // if we have a single SVP, and one plot, display the param name for that SVP
                    // on the xgrid above the plot, as to emulate a title, if one wants to:
                    if (singleValueParamNames.length == 1){
                        gridlabels2show.push(singleValueParamNames[0]);
                    }
                    
                    // assure SVPs are at least two:
                    while (singleValueParamNames.length < 2){
                        singleValueParamNames.push(addParam(false));
                    }
                    
                    // set the two SVPs as only choosable grid option (the option will be hidden as
                    // there is nothing else to choose):
                    paramNames = singleValueParamNames.slice(0, 2);
                    gridlayouts['---'] = paramNames;  // the key of gridlayouts is ininfluent
                    selectedgridlayout = '---';
                }else{
                    
                    // case 2 (see DEFINITIONS above): only N>=0 SVPs and the plots count is more than one:
                    // this should not happen but neverthless provide a 'stack horizontally' and
                    // 'stack vertically' grid options, by the order specified by building a "fake"
                    // multi-value-param which assigns an incremental index to each plot

                    // assure SVPs are at least one:
                    while (singleValueParamNames.length < 1){
                        singleValueParamNames.push(addParam(false));
                    }
                    // create MVP:
                    multiValueParamNames.push(addParam(true));
                    
                    // set the two combinations of the SVP and the MVP as only choosable grid options
                    // (providing a 'stack horizontally and 'stack vertically' generic names):
                    paramNames = [singleValueParamNames[0], multiValueParamNames[0]];
                    gridlayouts['&harr; stack horizontally'] = [paramNames[1], paramNames[0]];
                    gridlayouts['&varr; stack vertically'] = [paramNames[0], paramNames[1]];
                    selectedgridlayout = '&varr; stack vertically';
                }
            }else{
                
                // assure SVPs are at least two:
                while (singleValueParamNames.length < 2){
                    singleValueParamNames.push(addParam(false));
                }

                // take the multi value param and first single-value param:
                paramNames = multiValueParamNames.concat(singleValueParamNames[0]).concat(singleValueParamNames[1]);
                gridlabels2show = Array.from(multiValueParamNames);

                // always provide a grid option selecting a single plot: when chosen, the MVPs will be set
                // as this.selectedParams and displayed in one or more single <select> on top of the plots 
                gridlayouts['single plot'] = [singleValueParamNames[0], singleValueParamNames[1]];

                if (multiValueParamNames.length == 1){
                    
                    // case 3 (see DEFINITIONS above): N>=0 SVPs and 1 MVP: basically same as
                    // case 2 above but we do not need to create a fake multi-value param, 
                    // we use what we have providing the parameter name in the grid options
                    // (horizontal vs vertical) with the parameter name
                    // instead of generic 'stack horizontally' or 'stack vertically'

                    var svp = singleValueParamNames[0];
                    var mvp = multiValueParamNames[0];
                    gridlayouts[`&harr; ${mvp}`] = [mvp, svp];
                    gridlayouts[`&varr; ${mvp}`] = [svp, mvp];
                    selectedgridlayout = `&varr; ${mvp}`;
                }else{
                    
                    // case 4 (see DEFINITIONS above): N>=0 SVPs and M>1 MVPs: build a choosable grid
                    // with all combinations of the given MVPs times 2, as for each couple of P1, P2
                    // we can display the grid as P1xP2 or P2xP1
                    
                    for (var prm1 of multiValueParamNames){
                        for (var prm2 of multiValueParamNames){
                            if (prm1 === prm2){
                                continue;
                            }
                            gridlayouts[`&harr; ${prm1} vs. &varr; ${prm2}`] = [prm1, prm2];
                            if (!selectedgridlayout){ // take the first combination as selected one:
                                selectedgridlayout = `&harr; ${prm1} vs. &varr; ${prm2}`;
                            }
                        }
                    }
                }
            }
            // replace sets with sorted Arrays (vuejs wants Arrays, not Sets), and remove proeprties that are 'empty'
            // (either empty array or have a single falsy element (empty string, null)):
            var params = {};
            for(var key of paramNames){
                var values = Array.from(allparams[key]);
                // now sort values. Note that javascript array sort converts to string first (WTF!!!??), so
                // use a particular sort function for numbers only (see https://stackoverflow.com/a/1063027):
                if (typeof values[0] === 'number'){
                    values.sort((a, b) => a - b);
                }else{
                    values.sort();
                }
                params[key] = values;
            }
            // set defaults:
            this.gridlayouts = gridlayouts;
            this.selectedgridlayout = selectedgridlayout;
            this.paramnames2showongrid = new Set(gridlabels2show);
            this.params = params;
            // update grid layouts:
            this.gridLayoutChanged();
        },
        gridLayoutChanged: function(){
            var params = this.params;
            var [gridx, gridy] = this.gridlayouts[this.selectedgridlayout];
            var selectedParams = {};
            for(var paramName of Object.keys(params)){
                if (paramName == gridx || paramName == gridy || params[paramName].length <= 1){
                    continue;
                }
                // if value is in this.selectedParams, keep that value, otherwise
                // take the first one:
                var val = (paramName in this.selectedParams) ? this.selectedParams[paramName] : params[paramName][0];
                // set as selected param the first value:
                selectedParams[paramName] = val;
            }
            this.selectedParams = selectedParams;
        },
        setupAxisDefaults(){
        	// sets up the axis default for each plot and in the VueJs settings
        	// modifies each plot xaxis / yaxis data and `this.axis` Object
        	//
        	// Remember that:
        	// 1. `plots` is a subset of `this.plots` and it is an Array of plots,
        	// i.e. Objects of the form:
        	// plot= {
        	//		traces: [] //list of Plotly Objects representing traces
        	//		xaxis: {} //Plotly Object representing x axis
        	//		yaxis: Plotly Object representing y axis
        	//	 }, ..
        	//
        	// The Vue Object controlling axis settings is of the form:
        	// axis: {
            //    		x: {
            //   			log: {value:undefined, disabled:false},  // undefined: infer from plots, null: disable the option, otherwise boolean
            //   			sameRange: {value:undefined, disabled: false} // same as above
            //   		},
  			//		y: { same as above }
            //   }, 
        	
        	var keys = ['axis.x.log.value',
            		 	'axis.y.log.value',
            		 	'axis.x.sameRange.value',
            		 	'axis.y.sameRange.value'];
        	this.watchOff(...keys);
        	
        	// Set "private" variable in the xaxis and yaxis Objects used to
        	// remember some defaults, if set.
        	// Currently, we set _range and _type, which are the counterparts
        	// of 'range' and 'type', respectively. The latter are Plotly reserved
        	// keyswords and apparently it's harmless to add our custom (underscored)
        	// counterparts
        	this.plots.forEach(plot => {
        		if (!plot.xaxis){
        			plot.xaxis={};
        		}
        		if (!plot.yaxis){
        			plot.yaxis={};
        		}
        		var [rangex, rangey] = this.getPlotBounds(plot.traces || []);  // note that hidden traces count
        		plot.xaxis._range = rangex;
        		plot.yaxis._range = rangey;
        		
        		plot.xaxis._type = !(plot.xaxis.type) || (plot.xaxis.type == 'log') ? '' : plot.xaxis.type;
        		plot.yaxis._type = !(plot.yaxis.type) || (plot.yaxis.type == 'log') ? '' : plot.yaxis.type;
        	});

        	// setup default values:
        	for (var key of ['x', 'y']){
            	var axis = this.axis[key];  // the key for this,.axis ('x' or 'y')
            	var plotkey = key + 'axis';  //  the key for each plot axis: 'xaxis' or 'yaxis'
            	// axis.sameRange.disabled = false;
            	axis.sameRange.value = false;
            	// get if all axis have the same bounds:
            	var pltBounds = this.plots[0][plotkey]._range;
            	if (this.plots.every(plot => plot[plotkey]._range.every((elm, index) => elm === pltBounds[index]))){
            		// axis.sameRange.disabled = true;
            		axis.sameRange.value = true;
            	}
            	// log scale:
            	axis.log.value = this.plots.every(plot => plot[plotkey].type === 'log');
            	// set log scale disabled if the traces do not have 'log' as type and there are axis bound <=0:
            	//axis.log.disabled = !axis.log.value && plots.some(plot => plot[plotkey]._range.some(n => isNaN(n) || n <= 0));
            }

            // restart watching: 
            for (var key of keys){
            	// watch each prop separately because with 'deep=true' react is called more than once ...
            	this.watchOn(key, function(newval, oldval){
                	this.react();
            	});
            }
        },
        newPlot: function(divId){
            /**
             * Filters the plots to display according to current parameters and grid choosen, and
             * calls Plotly.newPlot on divId (the id of the chosen <div>)
             * divId undefined (not passed) will use the default div Id, i.e. draw on this component's plots <div>
             */
            if (divId === undefined){
                divId = this.plotdivid;
            }
            this.$nextTick(() => {
            	var [hover, drag] = ['mouseMode.hovermode', 'mouseMode.dragmode'];
            	this.watchOff(hover, drag);
                var [data, layout] = this.createPlotlyDataAndLayout(divId);
                Plotly.purge(divId);  // Purge plot. FIXME: do actually need this?
                Plotly.newPlot(divId, data, layout, this.defaultplotlyconfig);
                this.watchOn(hover, function (newval, oldval) {
                    this.setMouseModes(newval, undefined);  // hovermode, mousemode
                });
                this.watchOn(drag, function(newval, oldval){
                    this.setMouseModes(undefined, newval);  // hovermode, mousemode
                });
                
            });
        },
        react: function(divId){
            /**
             * Same as this.newPlot above, and can be used in its place to create a plot,
             * but when called again on the same <div> will update it far more efficiently
             */
            if (divId === undefined){
                divId = this.plotdivid;
            }
            this.$nextTick(() => {
                var [data, layout] = this.createPlotlyDataAndLayout(divId);
                // Purge plot. Check if we do actually need this:
                Plotly.react(divId, data, layout);
            });
        },
        createPlotlyDataAndLayout: function(divId){
            var plots = this.plots;
            var params = this.params;
            
            // filter plots according to selectedParams keys, i.e. the parameter names
            // which are mapped to mupliple values AND are not currently set to be displayed on the
            // grid x or grid y (basically, the parameters which will show up in a combo box on top of the whole plot)
            // The number of these parameters might be zero: in this case the filter below has no effect and we
            // still have all plots after the loop. Note also that the filter returns a copy of the original array.
            for (var key of Object.keys(this.selectedParams)){
                var val = this.selectedParams[key];
                plots = plots.filter(plot => plot.params[key] == val);
            }

            // console.log('creating plots');
     		this.setupPlotAxisData(plots);
			
            var [gridxparam, gridyparam] = this.gridlayouts[this.selectedgridlayout];
            var gridxvalues = params[gridxparam];
            var gridyvalues = params[gridyparam];
            var gridXval2index = new Map(gridxvalues.map((elm, idx) => [elm, idx]));
            var gridYval2index = new Map(gridyvalues.map((elm, idx) => [elm, idx]));
            // now build an array the same length as plots with each element the grids position [index_x, index_y]
            var plotsGridIndices = [];
            for (var plot of plots){
                var plotXGridIndex = gridXval2index.get(plot.params[gridxparam]);
                var plotYGridIndex = gridYval2index.get(plot.params[gridyparam]);
                plotsGridIndices.push([plotXGridIndex, plotYGridIndex]);
            }

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
            // copy layout annotations cause it's not a "primitive" type and thus we want to avoid old annotation to be re-rendered
            layout.annotations = layout.annotations ? Array.from(this.defaultlayout.annotations) : [];
            
            var data = [];
            var xdomains = new Array(gridxvalues.length);  // used below to place correctly the x labels of the GRID
            var ydomains = new Array(gridyvalues.length);  // used below to place correctly the x labels of the GRID
            var annotation = this.getAnnotation;
            var legendgroups = new Set();
            for (var i = 0; i < plots.length; i++){
                var [plot, [gridxindex, gridyindex]] = [plots[i], plotsGridIndices[i]];
                var [axisIndex, xaxis, yaxis, xdomain, ydomain] = this.getAxis(divId, gridyindex, gridxindex, gridyvalues.length, gridxvalues.length);
                xdomains[gridxindex] = xaxis.domain;  // used below to place correctly the x labels of the GRID
                ydomains[gridyindex] = yaxis.domain;  // used below to place correctly the y labels of the GRID
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
                // wrtie x and y labels: we might simply set xaxis.title= 'my label' (same for y axis) but:
                // 1. plotly reserves too much space between the axis line and the axis label (the space is out of control)
                // 2. Plotly does some hard coded calculation and moves the labels, and we want control over it:
                // for instance when two ploits are stacked vertically,
                // the bottom one has the legend moved up to be included in the plot area)
                
                // So, get xaxis.title and yaxis.title (if any) and put them as annotations:
                if (xaxis.title){
                    layout.annotations.push(annotation({
                        x: (xaxis.domain[1]+xaxis.domain[0])/2,
                        y: ydomain[0],
                        xanchor: 'center',
                        yanchor: 'bottom',
                        text: xaxis.title
                    }));
                    // delete xaxis title otherwise plotly draws the label:
                    delete xaxis.title;
                }
                // and ylabel: (why this? can't I set the tile to the xaxis object?)
                if (yaxis.title){
                    layout.annotations.push(annotation({
                        x: xdomain[0],
                        y: (yaxis.domain[1]+yaxis.domain[0])/2,
                        xanchor: 'left',
                        yanchor: 'middle',
                        textangle: '-90',
                        text: yaxis.title
                    }));
                    // delete xaxis title otherwise plotly draws the label:
                    delete yaxis.title;
                }
          }
          // Grid X labels: (horizontally on top)
          if (this.displayGridLabels_('x', gridxparam)){
              // determine the maximum y of all plot frames:
              // var y = Math.max(...ydomains.map(elm => elm[1]));
              // create the x labels of the vertical grid:
              for (var [domain, gridvalue] of xdomains.map((elm, index) => [elm, gridxvalues[index]])){
                  layout.annotations.push(annotation({
                      x: (domain[1] + domain[0])/2,
                      y: 1,
                      xanchor: 'center', /* DO NOT CHANGE THIS */
                      yanchor: 'top',
                      text: `${gridxparam}: ${gridvalue}`
                }));
              }
          }
          // Grid Y labels: (vertically on the right)
          if (this.displayGridLabels_('y', gridyparam)){
              // determine the maximum x of all plot frames:
              // var x = Math.max(...xdomains.map(elm => elm[1]));
              // create the y labels of the vertical grid:
              for (var [domain, gridvalue] of ydomains.map((elm, index) => [elm, gridyvalues[index]])){
                  layout.annotations.push(annotation({
                      x: 1,
                      y: (domain[1] + domain[0])/2,
                      xanchor: 'right',
                      yanchor: 'middle', /* DO NOT CHANGE THIS */
                      text: `${gridyparam}: ${gridvalue}`,
                      textangle: '-270'
                }));
              }
          }
          return [data, layout];
        },
        setupPlotAxisData(plots){
        	// sets up the plotly axis data on the plots to be plotted, according to
        	// the current axis setting and the plot data

        	// set the plot.xaxis.type (same for y):
            for (var key of ['x', 'y']){
            	var axis = this.axis[key];  // the key for this,.axis ('x' or 'y')
            	var plotkey = key + 'axis';  //  the key for each plot axis: 'xaxis' or 'yaxis'
            	plots.forEach(plot => {
            		plot[plotkey].type = axis.log.value ? 'log' : plot[plotkey]._type;
            		// we might have falsy plot type, meaning that the non-log type was unspecified
            		// thus remove:
            		if (!plot[plotkey].type){
	            		delete plot[plotkey].type;  // reset to default, whatever it is
	            	}
            	});
            	// set disabled state (some plots are not displayable as log. Although they might be actually be
            	// displayed as log, this depends on the default settings, just prevent here the user to switch log on/off):
	            axis.log.disabled = !axis.log.value && plots.some(plot => plot[plotkey]._range.some(n => isNaN(n) || n <= 0));
            }

			// set the plot.xaxis.type (same for y):
			var [sign, log10] = [Math.sign, Math.log10];
			// displaybars tells us if we are displaying (at least one) bar: this changes the behaviour
			// of the margin for custom-defined ranges (see below)
			var displaybars = plots.every(plot => plot.traces.some(trace => trace.type === 'bar'));
            for (var key of ['x', 'y']){
            	var axis = this.axis[key];  // the key for this,.axis ('x' or 'y')
            	var plotkey = key + 'axis';  //  the key for each plot axis: 'xaxis' or 'yaxis'
            	
				// set same Range disabled, preventing the user to perform useless click:
            	// Note that this includes we are shoing only one plot:
            	var pltBounds = plots[0][plotkey]._range;
            	axis.sameRange.disabled = 
            		(plots.every(plot => plot[plotkey]._range.every((elm, index) => elm === pltBounds[index])));
            	if (!axis.sameRange.value || axis.sameRange.disabled){
            		plots.forEach(plot => {
	        			delete plot[plotkey].range;
	        		});
            		continue;
            	}
            	// here deal with the case we have 'sameRange' clicked (and enabled):
            	var range = [NaN, NaN];
            	plots.forEach(plot => {
            		range = this.nanrange(...range, ...plot[plotkey]._range);
            	});
            	// calculate margins. Do not add them IF:
            	// 1. log scale and adding/removing margin causes value <=0
            	// 2. if range[0], Is displaybars, range  if: no log displayed or bound+- margin still positive,
            	// no bar displayed OR signum min
            	var margin = Math.abs(range[1] - range[0]) / 50;
            	if (displaybars && sign(range[0]) >= 0 && sign(range[1]) >= 0){
            		// do not add margin to the minimum
            	}else if (!axis.log.value || range[0] - margin > 0){
	        		range[0] -= margin;
	        	}
	        	if (displaybars && sign(range[0]) <= 0 && sign(range[1]) <= 0){
            		// do not add margin to the maximum
            	}else if (!axis.log.value || range[1] + margin > 0){
	        		range[1] += margin;
	        	}
	        	if (range.every(elm => !isNaN(elm))){
	        		plots.forEach(plot => {
	        			// plotly wants range converted to log if axis type is 'log':
	        			plot[plotkey].range = plot[plotkey].type === 'log' ? [log10(range[0]), log10(range[1])] : range;	
	            	});
            	}
           	}
        },
        displayGridLabels_(axis, paramName){
            if (this.paramnames2showongrid.has(paramName)){
                return this.displayGridLabels(axis, paramName, this.params[paramName]);
            }
            return false;
        },
        getAxis: function(divId, row, col, rows, cols){
            // computes the sub-plot area according to the row and col index
            // returns the array [axisIndex, xaxis, yaxis, xdomain, ydomain]
            // where xaxis and yaxis are the Objects to be passed to plotly's layout, xdomain = [x1, x2] and
            // ydomain = [y1, y2] are the two-element arrays denoting the enclosing area of the sub-plot
            var [uwidth, uheight] = this.getEmUnits(divId, this.plotfontsize);
            var [gridxparam, gridyparam] = this.gridlayouts[this.selectedgridlayout];
            if (!this.displayGridLabels_('x', gridxparam)){
                gridxparam = '';
            }
            if (!this.displayGridLabels_('y', gridyparam)){
                gridyparam = '';
            }
            var tt = gridxparam ? 1.5 * uheight : 0;
            var rr = gridyparam ? 3 * uwidth : 0;
            // the legend, if present, is not included in the plot area, so we can safely ignore it. Comment this line:
            // rr += 0. * uwidth * Math.max(...Object.keys(this.plotTraceColors).map(elm => elm.length)) ;
            var axisIndex = 1 + row * cols + col;
            // assure the dimensions are at least a minimum, otherwise plotly complains (assuming 10px as font-minimum):
            var [minuwidth, minuheight] = this.getEmUnits(divId, 10);
            // calculate plot width and height:
            var colwidth = Math.max(minuwidth, (1-rr) / cols);
            var rowheight = Math.max(minuheight, (1-tt) / rows);
            // determine the xdomain [x0, x1] defining the enclosing plot frame width (including ylabel):
            var xdomain = [col*colwidth, (1+col)*colwidth];
            // determine the ydomain [y0, y1] defining the enclosing plot frame height (including xlabel):
            var ydomain = [(rows-row-1)*rowheight, (rows-row)*rowheight]; // (y coordinate 0 => bottom , 1 => top)
            // define now the plotly x and y domains, which do NOT include x and y labels. Define paddings:
            var b = 3 * uheight;
            var t = 1 * uheight;
            var l = 4.5 * uwidth;
            var r = 1 * uwidth;
            // now define domains:
            var xaxisdomain = [xdomain[0]+l, xdomain[1]-r];
            var yaxisdomain = [ydomain[0]+b, ydomain[1]-t];
            // check that the domains are greater than a font unit:
            if (xaxisdomain[1] - xaxisdomain[0] < minuwidth){
                xaxisdomain = xdomain;
            }
            // check that the domains are greater than a font unit:
            if (yaxisdomain[1] - yaxisdomain[0] < minuheight){
                yaxisdomain = ydomain;
            }
            var xaxis = {domain: xaxisdomain, anchor: `y${axisIndex}`};
            var yaxis = {domain: yaxisdomain, anchor: `x${axisIndex}`};
            //console.log('xdomain:' + xdomain); console.log('ydomain:' + ydomain);
            return [axisIndex, xaxis, yaxis, xdomain, ydomain];
        },
        getEmUnits: function(divId, fontsize){
            // returns [uwidth, uheight], the units of a 1em in percentage of the plot div, which must be shown on the browser
            // Both returned units should be < 1 in principle
            var fontsize = this.plotfontsize;
            var [width, height] = this.getPlotDivSize(divId);
            return [fontsize/width, fontsize/height];
        },
        getAnnotation(props){
            return Object.assign({
                xref: 'paper',
                yref: 'paper',
                showarrow: false,
                font: {size: this.plotfontsize}
          }, props || {});
        },
        getPlotDivSize: function(divId){
            var elm = document.getElementById(divId);
            return [elm.offsetWidth, elm.offsetHeight];
        },
        toggleTraceVisibility: function(traceName){
            var visible = null;
            // iterate over all plot traces and set the visible property to its opposite:
            for(var plot of this.plots){
                for (var trace of plot.traces){
                    if (trace.legendgroup === traceName){
                        if (visible === null){  // visible uninitialized, set it:
                            // trace might not have the visible property (meaning: true)
                            var traceVisibility = 'visible' in trace ? trace.visible : true;
                            visible = !traceVisibility;
                        }
                        trace.visible = visible;
                    }
                }
            }
            // now get the indices of the currently displayed plots to update them:
            var indices = [];
            var plotlydata = this.getPlotlyDataAndLayout()[0]; 
            plotlydata.forEach(function(data, i){
                if(data.legendgroup === traceName){
                    indices.push(i);
                }
            });
            if(indices.length){
                Plotly.restyle(this.plotdivid, {visible: visible}, indices);
            }
        },
        isTraceVisible: function(traceName){
            // get the first plotly trace with the legendgroup matching `traceName`
            // and check its 'visible' property. Note that missing property defaults to true,
            // so check for properties explicitly set as false:
            for(var plot of this.plots){
                for (var trace of plot.traces){
                    if (trace.legendgroup === traceName){
                        return 'visible' in trace ? trace.visible : true;
                    }
                }
            }
            return false;
        },
        getPlotlyDataAndLayout: function(){
            /**
             * returns [data, layout] (Array an Object),
             * i.e. the Plotly data and layout used to draw the plot
             */
            var elm = document.getElementById(this.plotdivid);
            return elm ? [elm.data || [], elm.layout || {}] : [[], {}];
        },
        getPlotBounds: function(traces){
        	// gets the bounds min max of all the traces of 'plot', which must be an
        	// element of this.plots. Returns [rangeX, rangeY] where each range has two numeric elements
        	// pmin, max] which might be NaN
        	var [rangeX, rangeY] = [[NaN, NaN], [NaN, NaN]];
        	for (var trace of traces){
        		var [rangex, rangey] = [this.nanrange(...trace.x), this.nanrange(...trace.y)];
        		rangeX = this.nanrange(...rangeX, ...rangex);
        		rangeY = this.nanrange(...rangeY, ...rangey);
        	}
        	return [rangeX, rangeY];
        },
        nanrange: function(...values){
			values = values.filter(elm => typeof elm === 'number' && !isNaN(elm));
		  	var m = Math;
		  	return values.length? [m.min(...values), m.max(...values)] : [NaN, NaN];
		},
        download: function(){
            
            var props = this.downloadopts;
            if (!props.format){
                return;
            }
            
            if (props.format.startsWith('json')){
            	this.downloadAsJson();
            }else{
	            // FIXME: 1. props.size = 5 seems to increase resolution (by simply increasing the size)
	            // However, the fonts and the lines should be increased, too
	            // 2. legend is not present. If we want to add a legend, better would be to do it automatically
	            // (only one trace VISIBLE => no legend, otherwise yes), but then consider plotting on a new div, where
	            // we temporarily set this.defaultlayout.showlegend=true (currently, by changing showlegedn on the fly
	            // (e.g. in setBackround below) raises
	            // 3. Font size is not preserved in the image. But this has lower priority, as it turns out the
	            // font image (at least on Chrome + MacOsX) is a default Helvetica-like font which might be preferable
	            // as very common and readable
	            
	
	            var elm = document.getElementById(this.plotdivid);
	            var props = Object.assign({}, props);  // Object.assign(target, ...sources);
	            var elm = document.getElementById(this.plotdivid);
	            var [width, height] = [elm.offsetWidth, elm.offsetHeight];
	            if (!(props.width)){
	                props.width = (props.height ? width*props.height/height : width); // + 'px';
	            }
	            if (!(props.height)){
	                props.height = (props.width ? height*props.width/width : height); // + 'px';
	            }
	
	            function setBackground(gd) {
	                // https://community.plot.ly/t/plotly-toimage-background-color/8099
	                
	                // this function actually allows the user to configure the plot
	                // before rendering it to imag (why is it called like this
	                // in plotly?) but note that not all modifications work as expected:
	                
	                // the paper bg color is set to transparent by default. However,
	                // jpeg does not support it and thus we would get black background.
	                // Therefore:
	                if(props.format == 'jpeg'){
	                    gd._fullLayout.paper_bgcolor = 'rgba(255, 255, 255, 1)';
	                }
	                
	                // this actually does not work (raises) if we change the showlegend:
	                // thus, more work is needed to setting showlegend=true temporarily and then
	                // drawing on a new div
	                // gd._fullLayout.showlegend = true;
	            }
	            props.setBackground = setBackground;
	
	            Plotly.downloadImage(elm, props);
	        }
            this.downloadopts.show = false;
            this.downloadopts.format = '';

//             // Plotly.toImage will turn the plot in the given div into a data URL string
//             // toImage takes the div as the first argument and an object specifying image properties as the other
//             Plotly.toImage(elm, {format: 'png', width: width, height: height}).then(function(dataUrl) {
//                 // use the dataUrl
//             })
        },
        downloadAsJson(){
        	var filename = this.downloadopts.filename;
		    var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(this.data));
		    var downloadAnchorNode = document.createElement('a');
		    downloadAnchorNode.setAttribute("href",     dataStr);
		    downloadAnchorNode.setAttribute("download", `${filename}.json`);
		    document.body.appendChild(downloadAnchorNode); // required for firefox
		    downloadAnchorNode.click();
		    downloadAnchorNode.remove();
		},
        addResizeListener: function(fireResizeNow){
        	// adds (if not already added) a resize listener redrawing plots on window.resize
        	if (!this.resizeListener){
        		 // see prop below                    
                // avoid refreshing continuously, wait for resizing finished (most likely):
            	var resizeTimer;
                var self = this;
                this.resizeListener = function(){
                    clearTimeout(resizeTimer);
                    resizeTimer = setTimeout(() => {
                        self.react();
                    }, 300);
                };
                window.addEventListener('resize', this.resizeListener);
                if (fireResizeNow){
                	this.react();
                }
            }
       	},
       	removeResizeListener: function(){
       		// removes (if it has been set) the resize listener redrawing plots on window.resize
        	if (this.resizeListener){
        		window.removeEventListener('resize', this.resizeListener);
        		this.resizeListener = null;
            }
       	},
       	setMouseModes: function(hovermode, dragmode){
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
       			Plotly.relayout(this.plotdivid, layout);
       		}
       	}
    },
    mounted: function() { // https://stackoverflow.com/questions/40714319/how-to-call-a-vue-js-function-on-page-load
        // no -op
    }
});