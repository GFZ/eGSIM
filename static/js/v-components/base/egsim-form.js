/* Form components */

/* Base class for Form components. See README.md for details */
var BASE_FORM = {
	props: {
		form: Object,  // field names mapped to Objects describing <input>s or <select>
					   // (the keys 'value' and 'error' are mandatory)
		url: String,  // the request URL after form submission
	},
	data() {
		return {}
	},
	methods: {
		submit(){
			// submits the form data at the form URL (see props).
			// This method can be used with a form template written as:
			// <form novalidate @submit.prevent="submit"> ...

			// send the main post request to `this.url` using `this.form` as POST data
			return this.post(this.url).then(response => {
				if (response && response.data){
					return response; // allows .then on the Promise
				}
				throw new Error('response empty');  // should allow to .catch the promise in case
			}).catch(response => {
				var errData = (response.response || {}).data;
				var error = (errData || {}).error || {};
				// set the data field errors
				var errors = error.errors || [];
				for (var err of errors){
					var paramName = err.location;
					for (attrName of Object.keys(this.form)){
						var formField = this.form[attrName];
						if (formField.name === paramName){
							formField.error = err.message || 'invalid: unknown error';
						}
					}
				}
				throw response;   // https://www.peterbe.com/plog/chainable-catches-in-a-promise
			});
		},
		post(url){
			// send a post request to the given url using this.form as POST data. Returns
			// a `Promise` object that can be chained (see e.g. `this.request`)
			for (var key of Object.keys(this.form)){  // clear errors first
				this.form[key].error = "";
			}
			var [data, config] = this.getPostDataAndConfig();
			return this.$httpClient.post(url, data, config);
		},
		getPostDataAndConfig(){
			// Returns the arguments for a POSt request in the form of the Array
			// [postData, config], where data is either an Object or a
			// FormData object, and config is an Object with varying keys depending on
			// the data of `this.form` (basically: this.form has files or not)
			var form = this.form;
			var hasFiles = Object.keys(form).some(elm => form[elm].value instanceof File);
			if (hasFiles){
				const config = {
					headers: {
						'content-type': 'multipart/form-data'
					}
				};
				return [this.formToFormData(), config];
			}
			return [this.formToJSON(), {}];
		},
		formToJSON(){
			// Return `this.form` as JSON serializable Object
			data = {};
			for (var key of Object.keys(this.form)){
				var field = this.form[key];
				if (!field.disabled){
					// assign value to object up to be sent, use the "name" as key:
					data[field.name] = field.value;
				}
			}
			return data;
		},
		formToFormData(){
			// snippet from:
			// https://www.codegrepper.com/code-examples/javascript/axios+upload+a+multipart%2Fform-data
			const formData = new FormData();
			var formObj = this.formToJSON();
			for (var name of Object.keys(formObj)){
				// https://stackoverflow.com/a/63340869:
				// in form-data content, null will be converted to "null". Either replace
				// (but how to get a default?) or simply remove:
				var val = formObj[name];
				if ((val !== null) && (val !== undefined)){
					formData.append(name, val);
				}
			}
			return formData;
		},
	}
};

/**Egsim form used in trellis, residuals, testing. Main features:
- emits a 'submitted' on response successfully received, after submit
- implements a toolbar for IO operations such as get Form in YAML or JSON config
- Deals with hiding and transforming the form into a dialog popup after first submit*/
EGSIM.component('egsim-form', {
	mixins: [BASE_FORM],  // will have props Form, url, and all methods for issuing post requests
	props :{
		downloadUrl: String,  // url for downloading the current form as config yaml/json
		visibilityToggle: {type: Boolean, default: true},  // variable to toggle form visibility from external components
	},
	data() {
		return {
			show: true,
			showAsDialog: false,  // appearance control
			mounted: false,  // needed for ? FIXME
			idRequestURLInput: this.url + '_requesturl_input_',
			requestURL: '',
			watchers: [],  // needed for FIXME
			downloadActions: this.createDownloadActions()
		}
	},
	emits: ['submitted'], // Vue 3 required attr (in case we migrate)
	methods: {
		submitMe(responseData){  // overrides super method
			this.submit().then(response => {
				this.show = !this.show;
				this.showAsDialog = true;
				setTimeout(() => {
					// notify asynchronously after the form has been hidden:
					this.$emit('submitted', response);
				}, 250);
			});
		},
		// toolbar methods:
		fetchRequestURL(){
			// Fetches
			// the current config (request) as dict and builds this.requestURL
			// Returns an axios Promise to which a user can attach functions to
			// be executed when the POST request returns successfully

			// *NOTE*: in Chrome, after clicking on the button calling this function,
			// when we move out of it, the tooltip stays there: to make it disappear,
			// we need to focus something else. This is annoying but we could not fix it
			// (we tried implementing a wrapper method, which was hiding the aria-label
			// and then restoring it later inside a `then` attached to the returned promise
			// below). If you want the source button, pass src as argument and access
			// src.currentTarget
			return this.post(this.downloadUrl + '.json').then(response => {
				if (response && response.data){
					var responseData = response.data;
					var retUrl = window.location.origin;
					if (!window.location.origin) {  // https://stackoverflow.com/a/25495161
						retUrl = window.location.protocol + "//"
							+ window.location.hostname
							+ (window.location.port ? ':' + window.location.port : '');
					}
					retUrl += (this.url.startsWith('/') ? '' : '/') + this.url;
					var prefix = '?';
					for (var paramName of Object.keys(responseData)){
						retUrl += `${prefix}` + encodeURIComponent(paramName) + '=' + encodeURIComponent(responseData[paramName]);
						prefix = '&';
					}
					this.watchForValueChanges(true);
					this.requestURL = retUrl;
				}
			});
		},
		watchForValueChanges(watch){
			if (watch == !!this.watchers.length){
				return;
			}
			if (watch){
				for (var key of Object.keys(this.form)){
					this.watchers.push(this.$watch(`form.${key}.value`, (newVal, oldVal) => {
						this.requestURL ='';
						this.watchForValueChanges(false);
					}));
				}
				return;
			}
			// unwatch: simply call the stored callback
			this.watchers.forEach(wacther => wacther());
			this.watchers = [];
		},
		copyRequestURL(src){
			var targetElement = src.currentTarget; // the button
			// aria-label is used by balloon.css to display the tooltip
			// store the current one:
			var ariaLabel = targetElement.getAttribute('aria-label');
			var successful = this.copyText(document.getElementById(this.idRequestURLInput));
			// restore the old aria-label
			if (ariaLabel){
				targetElement.setAttribute('aria-label', successful ? 'Copied' : 'Unable to copy');
				setTimeout(() => {
					targetElement.setAttribute('aria-label', ariaLabel);
				}, 1000);
			}
		},
		copyText(element){
			// https://www.w3schools.com/howto/howto_js_copy_clipboard.asp
			element.focus();
			element.select();
			try {
				var successful = document.execCommand('copy');
			} catch (err) {
				var successful = false;
			}
			return successful;
		},
		readLocalJSON(src){
			// reads a local uploaded file from src.currentTarget
			// copied and modified from http://researchhubs.com/post/computing/javascript/open-a-local-file-with-javascript.html
			var fileInput = src.currentTarget;
			var file = fileInput.files[0];
			if (!file) {
				return;
			}
			var form = this.form;
			var self = this;
			var reader = new FileReader();
			reader.onload = function(e) {
				var contents = e.target.result;
				// clear the file value otherwise when clicking again on the same file
				// we do not get the change event:
				// setting an empty string seems not to call again the change event
				// but in any case this method should return immediately (see if above)
				fileInput.value = "";
				// Display file content
				var obj = {};
				try{
					var obj = JSON.parse(contents);
				}catch(error){
					// although discouraged, this.$root is the easiest way to notify the main root
					// and display the error:
					self.$root.setError('Invalid file. Check that you loaded a JSON-formatted text file');
				}
				for (var key of Object.keys(obj)){
					if (!(key in form)){
						self.$root.setError(`Invalid JSON. "${key}" is not a valid parameter`);
						return;
					}
				}
				self.resetDefaults.call(self);
				for (var key of Object.keys(obj)){
					form[key].value = obj[key];
				}
			};
			reader.readAsText(file);
		},
		createDownloadActions(){
			return ['json', 'yaml'].map(ext => {
				var url = this.downloadUrl + "." + ext;
				return [ext, () => {
					this.$httpClient.download(url, this.formToJSON()); // see egism_base.js
				}];
			}, this);  // <- make `this` in `map` point to this Vue instance
		}
	},
	mounted() {
		// set the mounted variable in order to activate the transition after the
		// whole component has been mounted
		// the transition is used just to show up / hide the form
		if (this.mounted){
			return;
		}
		this.$nextTick(function () {
			// Code that will run only after the
			// entire view has been rendered
			this.mounted = true;
		});
	},
	watch: {
		visibilityToggle(newVal, oldVal){
			this.show = !this.show;
		}
	},
	template: `<form novalidate @submit.prevent="submitMe"
		  class="flex-column position-relative pb-4 align-self-center"
		  :class="[showAsDialog ? ['shadow', 'border', 'bg-body', 'mt-1', 'mb-3'] : '']"
		  style="flex: 1 1 auto;z-index:10; border-color:rgba(0,0,0,.3) !important"
		  :style="{'display': show ? 'flex' : 'none'}">

		<div class="d-flex flex-column" style="flex: 1 1 auto">

			<div class='d-flex flex-row justify-content-center align-items-baseline p-1 mb-3 bg-light'>

				<button type="button" onclick='this.nextElementSibling.click()'
						data-balloon-pos="down" data-balloon-length="medium"
						aria-label="Load a configuration from a local JSON-formatted text file. This can be, e.g., a configuration previously saved as JSON (see 'Download as')"
						class="btn btn-outline-dark border-0">
					<i class="fa fa-upload"></i>
				</button>
				<!-- NOTE: the control below must be placed immediately after the control above! -->
				<input style='display:none' type="file" id="file-input" @change='readLocalJSON'>

				<action-select :actions="downloadActions" style='width:initial !important'
							   class="ms-2 form-control form-control-sm bg-transparent border-0"
							   data-balloon-pos="down" data-balloon-length="medium"
							   aria-label="Download the current configuration as text file. The file content can then be used in your custom code as input to fetch data (see POST requests in the API documentation for details)">
					Download as:
				</action-select>

				<button type="button" @click='fetchRequestURL'
						data-balloon-pos="down" data-balloon-length="medium"
						aria-label="Show the API URL of the current configuration. The URL can be used in your custom code to fetch data (see GET requests in the API documentation for details). You can also paste it in the browser to see the results (e.g., Firefox nicely displays JSON formatted data)"
						class="btn btn-outline-dark border-0 ms-2">
					<i class="fa fa-link"></i>
				</button>

				<input :id="idRequestURLInput" type='text' v-model='requestURL'
					   :style= "requestURL ? {} : { 'visibility': 'hidden'}"
					   class="form-control form-control-sm ms-2 bg-transparent border-0"
					   style="flex: 1 1 auto;width:initial !important"/>

				<button type="button" v-show='requestURL' @click="copyRequestURL"
						aria-label="Copy the URL" data-balloon-pos="down" data-balloon-length="medium"
						class="btn btn-outline-dark border-0">
					<i class="fa fa-copy"></i>
				</button>

				<button type="button" v-show='showAsDialog' @click="show=!show"
						aria-label="Close form window" data-balloon-pos="down" data-balloon-length="medium"
						class="btn btn-outline-dark border-0 ms-2">
					<i class="fa fa-times"></i>
				</button>

			</div>

			<div class="d-flex flex-row" :class="[showAsDialog ? ['mx-4'] : '']"
				 style="flex: 1 1 auto">
				<div class="d-flex flex-column" style="flex: 1 1 auto">
					<slot name="left-column"></slot>
				</div>

				<div class="d-flex flex-column ms-4" style="flex: 1 1 auto">

					<slot name="right-column"></slot>

					<div class='d-flex flex-row justify-content-center mt-4'>
						<div style='flex: 1 1 auto'></div>
						<button type="submit" class="btn btn-primary">
							<i class="fa fa-play"></i> Display results
						</button>
					</div>
				</div>
			</div>
		</div>
	</form>`
});


/* A <select> that performs an action for each <option> clicked, and first option
 disabled acting as a title (see <slot> in template)*/
EGSIM.component('action-select', {
	props: {
		// provide an Array of [caption, callback] pairs associated to each <option>:
		actions: {type: Array, default: () => { return []; }},
	},
	data() {
		var noAction = null;
		return {
			noAction: noAction,
			selectedAction: noAction
		}
	},
	computed: {
		options(){
			return this.actions.filter(elm => {
				return !!elm[0] && typeof elm[1] === 'function';
			});
		}
	},
	watch: {
		'selectedAction': function (newVal, oldVal){
			for (var [label, callback] of this.options){
				if (label === newVal){
					callback();
					break;
				}
			}
			this.selectedAction = this.noAction;
		}
	},
	template: `<select v-model='selectedAction'>
			<option :value='noAction' :disabled="true">
				<slot></slot>
			</option>
			<option v-for='[key, callback] in actions' :value='key'>
				{{ key }}
			</option>
		</select>`,
	methods: {
	}
});