/** base skeleton implementation for the base Vue instance. Also implements
 Vue.post global function (see page bottom)
 */
var EGSIM_BASE = {
    data: function(){ return {
        // NOTE: do not prefix data variable with underscore: https://vuejs.org/v2/api/#data
        loading: false,
        errormsg: '',
        selComponent: '',
        // FIXME: IMPLEMENT PROPS FOR DEFAULT VARIABLES?
        // gsims: {}, // an Object of gsim names (string) mapped to [[... imts ...], trt, oq_gsim_warning] (all elements strings)
        componentProps: {}, // an object of Objects keyed by each string denoting a component name (<=> menu tab)
        postfuncDefaultConfig: {}  // default config used in Vue.post. FIXME: better doc
        // In case we want to use an event bus:
        // https://laracasts.com/discuss/channels/vue/help-please-how-to-refresh-the-data-of-child-component-after-i-post-some-data-on-main-component/replies/288180
    }},
    created: function(){
        // Use regular expression to convert Gsim names to readable names by
        // splitting into space-separated tokens (in the following order of importance):
        // [A-Z]+[a-z]+ e.g. "Akkar" "BNCLower"
        // (?<!_)[0-9]+(?!_)  e.g. "2006" but not "2006_"
        // [A-Z_]+ or [a-z_]+ e.g. "ABC" "abc"
        // .* anything else returned as single token, e.g., "b2006_g"
        //var reg = /[A-Z]+[a-z]+|(?<!_)[0-9]+(?!_)|[A-Z_]+|[a-z_]+|.+/g;
        var reg = /[A-Z]+[^A-Z0-9]+|[0-9]+|.+/g;
        // converts the gsims received from server from an Array of Arrays to an
        // Array of Objects:
        var imts = [];
        var gsimObjects = this.gsims.map(elm => {
            var [gsimName, imts_, warning] = elm;
            // add imt:
            imts_.map(imt => {
                if (!imts.includes(imt)){
                    imts.push(imt);
                }
            });
            return {
                value: gsimName,
                disabled: false,
                // innerHTML (the display name) is gsimName with spaces: split
                // according to Camel Case words, or numbers, the rest (.*)
                // keep it together:
                innerHTML: gsimName.match(reg).join(" "),
                imts: imts_,
                warning: warning || "",
            }
        });
        var regionalization = this.regionalization;
        // set processed data:
        for (var [name, form] of this.forms()){
            if (form.gsim){
                // set form.gsim.choices as a deep copy of gsimObjects:
                form.gsim.choices = gsimObjects.map(elm => Object.assign({}, elm));
                form.gsim.regionalization = {
                    url: regionalization.url,
                    choices: regionalization.names.map(elm => [elm, elm]),
                    value: Array.from(regionalization.names)
                }
            }
            if (form.imt){
                // set form.imt as a deep copy of imts:
                form.imt.choices = Array.from(imts);
            }
        }
        // in `vueutils.js` we defined a POST function which emits the following events
        // on this instance. Create the POST function and event notifiers attached to this object
        Vue.createPostFunction(this, this.postfuncDefaultConfig);
        // and now listen for the just created event notifiers:
        this.$on('postRequestStarted', this.postRequestStarted);
        this.$on('postRequestCompleted', this.postRequestCompleted);
        this.$on('postRequestFailed', this.postRequestFailed);
        this.$on('postRequestEnded', this.postRequestEnded);
    },
    computed: {
        selComponentProps(){  // https://stackoverflow.com/a/43658979
            return this.componentProps[this.selComponent];
        }
    },
    mounted: function() {
        // https://stackoverflow.com/questions/40714319/how-to-call-a-vue-js-function-on-page-load
        // no -op for the moment
    },
    methods: {
        setComponent(name){
            this.selComponent = name;
            this.setUrlInBrowser(name);
        },
        setUrlInBrowser(menu){
            var location = window.location;
            if (!location.pathname.startsWith(`/${menu}`)){
                var newHref = `${location.origin}/${menu}`
                // https://developer.mozilla.org/en-US/docs/Web/API/History_API
                window.history.replaceState({}, document.title, newHref);
            }
            return false; // in case accessed from within anchors
        },
        /*
         * POST request listeners:
         */
        postRequestStarted(){
            this.setError('');
            this.setLoading(true);
        },
        postRequestCompleted(url, data, config, response){ /* no-op*/ },
        postRequestFailed(url, data, config, response){
            var errorMessage = response.message ||  'Unknown error';
            var errData = (response.response || {}).data;
            if (errData instanceof ArrayBuffer){
                // sometimes we might want to download png and json, we then
                // need to return an ArrayBuffer (see download.js).
                // The ArrayBuffer might "hide" a JSON formatted string. Thus,
                // try to convert to string and then json.parse:
                try{
                    // copied from: https://developers.google.com/web/updates/2012/06/How-to-convert-ArrayBuffer-to-and-from-String
                    // Uint8 because we send data as UTF8
                    errData = JSON.parse(String.fromCharCode.apply(null, new Uint8Array(errData)));
                }catch(exc){
                    errData = {};
                }
            }
            var error = (errData || {}).error || {};
            // set the data field errors:
            if (this.isFormObject(data)){
                var errors = error.errors || [];
                for (var err of errors){
                    var paramName = err.location;
                    if (paramName && (paramName in data)){
                        data[paramName].error = err.message || 'invalid: unknown error';
                    }
                }
            }
            // set the global error message:
            this.setError(error.message || errorMessage);
        },
        postRequestEnded(){
            this.setLoading(false);
        },
        /*
         * Components event handlers:
         */
        handleEmittedEvent(eventName){
            if (eventName == 'movetoapidoc'){
                this.moveToApidoc(arguments[1] || '');
            }else if (eventName == 'selectgsims'){
                this.selectGsims(arguments[1] || []);
            }
        },
        moveToApidoc(fragmentName){
            this.componentProps['apidoc'].fragment = fragmentName.startsWith('#') ? fragmentName : '#' + fragmentName;
            this.setComponent('apidoc');
        },
        selectGsims(gsims){
            for (var [name, form] of this.forms()){
                if (form.gsim){
                    form.gsim.val = Array.from(gsims);
                }
            }
        },
        /* other functions: */
        clearErrors(){
            this.setError('');
            // clear all errors in forms:
            // set processed data:
            for (var [name, form] of this.forms()){
                Object.keys(form).forEach(fieldname => {
                    form[fieldname].err = '';
                });
            }
        },
        forms(){
            var ret = [];
            Object.keys(this.componentProps).forEach(name => {
               var compProps = this.componentProps[name];
               if (typeof compProps === 'object'){
                   Object.keys(compProps).forEach(pname => {
                       var element = compProps[pname];
                       if (this.isFormObject(element)){  // defined in vueutil.js
                           ret.push([name, element]);
                       }
                   });
               }
            });
            return ret;
        },
        isFormObject(obj){
            // global function returning true if `obj` is a form Object, i.e. an Object where each of its
            // properties is a form field name (string) mapped to the form field (Object)
            if (typeof obj !== 'object'){
                return false;
            }
            // check if all form fields have two mandatory properties of the Object: val and err (there are more,
            // but the two above are essential)
            return Object.keys(obj).every(key => {
                var elm = obj[key];
                return (typeof elm === 'object') && ('value' in elm) && ('error' in elm);
            });
        },
        setError(error){
            this.errormsg = error;
        },
        setLoading(value){
            this.loading = value;
        }
    }
};


/**
 * Add global property / method / directives to Vue (https://vuejs.org/v2/guide/plugins.html)
*/
Vue.use({
    install : function (Vue, options) {
        Vue.isEmpty = function(obj){
            // global function returning true if `obj` is null, undefined or an empty Object
            return (obj === null) || (obj === undefined) || ((typeof obj === 'object') && Object.keys(obj).length === 0);
        };
        Vue.createPostFunction = function(root, defaultAxiosConfig){
            // creates a globally available POST function using axios, and notifying the
            // root instance. This function is called from egsim.html after creation of the main Vue instance
            Vue.post = (url, data, config) => {
                /**
                 * Perform a POST request. `Vue.post` can be called from any component and
                 * returns a promise which can be chained with .then(response) and .catch(response)
                 * where response is the axios response object. The function emits events on the root
                 * instance passed above, to control visual stuff (e.g., progress bar while fetching,
                 * display of errors)
                 *
                 * @param url: string of the url
                 * @param data: any data (usually Object) to be sent as POST body. This might include the "form" objects
                 *     in the form {field1: {err: '', val: V1, ... }, ..., fieldn: {err: '', val: Vn, ... }}
                 *     In this case, 1. the Object sent will be of the form {field1: V1, ... fieldn: Vn} and
                 *                   2. the fields errors ('err') will be set in case of form validation errors returned from the server
                 * @param config: any data (Object) for configuring the POST request
                 */
                // emit the starting of a POST:
                root.$emit('postRequestStarted');
                var config = Object.assign(config || {}, defaultAxiosConfig);  // Object.assign(target, source)
                // guess if we passed a form data object, and in case convert it to a JSONizable Object:
                var jsonData = data || {};
                return axios.post(url, jsonData, config).then(response => {
                    root.$emit('postRequestCompleted', url, data, config, response);
                    // allow chaining this promise from sub-components:
                    return response;  // https://github.com/axios/axios/issues/1057#issuecomment-324433430
                }).catch(response => {
                    root.$emit('postRequestFailed', url, data, config, response);
                    // allow chaining this promise from sub-components:
                    throw response;   // https://www.peterbe.com/plog/chainable-catches-in-a-promise
                }).finally(() => {
                    root.$emit('postRequestEnded');
                });
            }
        }
    }
});
