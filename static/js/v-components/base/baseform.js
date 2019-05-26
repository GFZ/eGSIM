/**
 * Represents a base form used in trellis, residuals, testing
 */
var _BASE_FORM = Vue.component('baseform', {
    props: {
        form: Object,
        url: String,
        response: {type: Object, default: () => {return {}}},
        post: Function,
        filename: {type: String, default: 'egsim'}
    },
    data: function () {
    	return {
        	responseDataEmpty: true,
            responseData: this.response
        }
    },
    methods: {
        request: function(){
            var form = this.form;
            this.post(this.url, form).then(response => {
                if (response && response.data){
                    this.responseData = response.data;
                } 
            });
        },
        download: function(filename, index, filenames){
        	var form = this.form;
        	var ext = filename.substring(filename.lastIndexOf('.')+1, filename.length);
            this.post("data/" + this.url + "/downloadrequest/" + filename, form).then(response => {
                if (response && response.data){
                    Vue.download(response.data, filename);
                } 
            });
        }
    },
    watch: {
        responseData: {
            immediate: true, // https://forum.vuejs.org/t/watchers-not-triggered-on-initialization/12475
            handler: function(newVal, oldVal){
                this.responseDataEmpty = Vue.isEmpty(newVal); // defined in vueutil.js
                if (!this.responseDataEmpty){
                	this.$emit('responsereceived', newVal);
                }
            }
        }
    },
    template: `
	<transition name="egsimform">
    <form novalidate v-on:submit.prevent="request"
        :class="[responseDataEmpty ? '' : ['shadow', 'border', 'bg-light']]"
        class='d-flex flex-column flexible position-relative mb-3 align-self-center' style='z-index:10'>
        
        <div class="d-flex flex-column flexible" :class="[responseDataEmpty ? '' : ['mx-4', 'mt-4', 'mb-3']]">
            <div class="d-flex flexible flex-row mb-3">

                <div class="d-flex flexible flex-column">
                    <gsimselect :form="form" showfilter class="flexible"></gsimselect>
                </div>
                
                <slot/> <!-- << HERE CUSTOM FORM ELEMENTS IN CHILD COMPONENTS -->

            </div>
        
			<div class='d-flex flex-row justify-content-center border-top pt-3'>
				<downloadselect
					:items="[filename + '.request.json', filename + '.request.yaml']"
					@selected="download"
				>
					Download request as:
				</downloadselect>
	            <button type="submit" class="btn btn-primary ml-2">
	                <i class="fa fa-play"></i> Display plots
	            </button>
	            <button type="button" class="btn btn-primary ml-2"
	            	v-show='!responseDataEmpty'
	            	@click='$emit("closebuttonclicked")'
	            >
	                <i class="fa fa-times"></i> Close
	            </button>
            </div>

        </div>
        
    </form>
	</transition>`
})