/**
 * Implements a component representing a form element (<input>, <select>)
 */
Vue.component('forminput', {
    //https://vuejs.org/v2/guide/components-props.html#Prop-Types:
    props: {
    	// form is an object where a name (string) is mapped to an Object representing HTML input elements (elm).
    	// Each elm is in turn an object with properties 'err' (string), 'val' (any),
    	// 'attrs' (Object of strings mapped to any value representing attributes dynamically bound to
    	// the HTML element via Vuejs), is_hidden (boolean), 'choices' (list of two element lists, for <select>s),
    	// label and help (both strings). Only 'err' and 'val' are mandatory. If you want to attach a new attribute
    	// to be bound to the element (e.g. `elm.attrs.disabled`) remember to use Vuejs $set to make the prop reactive
    	// before passing the form to this component (https://vuejs.org/v2/guide/reactivity.html#Change-Detection-Caveats)
        form: {type: Object},
        // the name of the input component. form[name] must be the Object 'elm' described above
        name: {type: String},
        // if true, display header only (name and infos) and not input element
        headonly: {type: Boolean, default:false},
        // if true, help button shows up in the header  and this component
        // emits an 'helprequested' event with no arguments (for the moment):
        showhelpbutton: {type: Boolean, default:false}
    },
    data: function () {
    	// define the Object whose data is associated to this component:
    	var elm = this.form[this.name];
    	var isSelect = elm.attrs.type == 'select' || (elm.choices && elm.choices.length);
        return {
        	elm: elm,
            // calculate (once) data for the template below:
            isCheckOrRadio: elm.attrs.type == 'radio' || elm.attrs.type == 'checkbox',
            // type might be undefined, thus we need to check the choices for <select>:
            isSelect: isSelect,
            isSelectMultiple: isSelect && elm.attrs.multiple
        }
    },
    template: `<div v-if="!elm.is_hidden">
        <div class="d-flex flex-row mb-0" :class="[showhelpbutton ? ['align-items-end'] : ['align-items-baseline']]">
            <label :for="elm.attrs.id" class='mb-0 text-nowrap' :class="[elm.attrs.disabled ? ['text-muted'] : ['font-weight-bold']]">
                <input v-if="!headonly && isCheckOrRadio" v-model="elm.val" v-bind="elm.attrs" class='mr-1'>
                <span v-html="elm.label"></span>
            </label>
            <div class="text-muted small flexible ml-3 text-right">
                <span v-if="elm.err" class="text-danger">{{ elm.err }}</span>
                <span v-else>[{{ name }}]</span>
            </div>
            <template v-if="!elm.err">
            	<div class="text-muted small ml-1" v-if="elm.help" v-html="elm.help"></div>
            	<div class="text-muted small ml-1" v-if="isSelectMultiple">{{ elm.val.length || 0 }} of {{ elm.choices.length }} selected</div>
            </template>
            <button v-if="showhelpbutton" type="button" @click='$emit("helprequested")'
    		 		class='btn btn-outline-secondary btn-sm ml-1 mb-1 py-0'>
    			<i class="fa fa-info-circle"></i>
    		</button>
        </div>
        <template v-if="!headonly">
	        <select
	        	v-if="isSelect"
	        	v-model="elm.val"
	        	v-bind="elm.attrs"
	        	class='form-control'
	        	:class="{'border-danger': !!elm.err}"
	        >
	        	<option
	        		v-for='opt in elm.choices'
	        		:value='opt[0]'
	        		v-html='getOptionLabel(opt)'>
	        	</option>
	    	</select>
	    	<input v-else-if="!isCheckOrRadio" v-model="elm.val" v-bind="elm.attrs" class='form-control'
	    		:class="{'border-danger': !!elm.err}">
    	</template>
    </div>`,
    methods: {
    	getOptionLabel: function(opt){
    		return opt[1] + (opt[0] == opt[1] ? '' : '&nbsp;&nbsp;[' + opt[0] + '] ')
    	}
    }
})