import React, { Component } from 'react';
import axios from 'axios';
import {Progress} from 'reactstrap';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';


var thumbnails = []

class ImgThumbnails extends React.Component {
  render() {
      if (thumbnails.length > 0)
//            return <img  style={{maxHeight:'90px', maxWidth:'70px'}} src={thumbnails[0]}/>;//src={this.state.file}/>;

            return (
            <div>
              {thumbnails.map((value, index) => {
                return <img  key={index} alt='your upload' style={{maxHeight:'135px', maxWidth:'105px', padding:'2px'}} src={value}/>
              })}
            </div>
          )


    return null;
  }
}



class App extends Component {
  constructor(props) {
    super(props);
      this.state = {
        selectedFile: null,
        loaded:0
      }

  }
  checkMimeType=(event)=>{
    //getting file object
    let files = event.target.files
    //define message container
    let err = []
    // list allow mime type
   const types = ['image/png', 'image/jpeg', 'image/gif']
    // loop access array
    for(let x = 0; x<files.length; x++) {
     // compare file type find doesn't matach
         if (types.every(type => files[x].type !== type)) {
         // create error message and assign to container
         err[x] = files[x].type+' is not a supported format\n';
       }
     };
     for(var z = 0; z<err.length; z++) {// if message not same old that mean has error
         // discard selected file
        toast.error(err[z])
        event.target.value = null
    }
   return true;
  }


  maxSelectFile=(event)=>{
      const MAX_FILES_UPLOAD = 5
      let files = event.target.files
      if (files.length > MAX_FILES_UPLOAD) {
         const msg = `Only ${MAX_FILES_UPLOAD} images can be uploaded at a time`
         event.target.value = null
         toast.warn(msg)
         return false;
      }
      return true;
  }

  checkFileSize=(event)=>{
      let files = event.target.files
      let size = 2000000
      let err = [];
      for(var x = 0; x<files.length; x++) {
          if (files[x].size > size) {
              err[x] = `${files[x].name} ${files[x].type} is too large (over 2mo), please pick a smaller file\n`;
          }
      };
      for(var z = 0; z<err.length; z++) {// if message not same old that mean has error
    // discard selected file
          toast.error(err[z])
          event.target.value = null
      }
      return true;
  }

  onChangeHandler=event=>{
      var files = event.target.files

      thumbnails = []

    // if return true allow to setState
      if (this.maxSelectFile(event) && this.checkMimeType(event) && this.checkFileSize(event)) {
            for (var i=0; i<files.length;i++)
                thumbnails.push(URL.createObjectURL(files[i]));

            this.setState({
                selectedFile: files,
                loaded:0
            })
        }
  }

  onClickHandler = () => {
      if (this.state.selectedFile === null || this.state.selectedFile.length === 0)
            toast.warning('please upload an image first')
      else {
          console.log(this.state.selectedFile)
          const data = new FormData()
          for(var x = 0; x<this.state.selectedFile.length; x++) {
              data.append('file', this.state.selectedFile[x])
          }
          axios.post("http://localhost:8000/upload", data, {
              onUploadProgress: ProgressEvent => {
                  this.setState({
                      loaded: (ProgressEvent.loaded / ProgressEvent.total*100),
                  })
              },
          })
          .then(res => { // then print response status
              toast.success('upload success')
          })
          .catch(err => { // then print response status
              toast.error('upload fail')
          })
      }
  }

  render() {
    return (
      <div className="container">
	      <div className="row">
      	  <div className="offset-md-3 col-md-6">
               <div className="form-group files">
                <label>Upload Your File </label>
                <input type="file" className="form-control" multiple onChange={this.onChangeHandler}/>

              </div>
              <ImgThumbnails />

              <div className="form-group">
              <ToastContainer />
              <Progress max="100" color="success" value={this.state.loaded} >{Math.round(this.state.loaded,2) }%</Progress>

              </div>

              <button type="button" className="btn btn-success btn-block" onClick={this.onClickHandler}>Upload</button>

	      </div>
      </div>
      </div>
    );
  }
}

export default App;


/*import React from 'react';
import './App.css';
import DragAndDrop from './DragAndDrop'



function ProgressHandler(e) {
	var complete = Math.round(e.loaded / e.total * 100);
	console.log(complete + "% complete");
}

function App() {

  const reducer = (state, action) => {
    switch (action.type) {
      case 'SET_DROP_DEPTH':
        return { ...state, dropDepth: action.dropDepth }
      case 'SET_IN_DROP_ZONE':
        return { ...state, inDropZone: action.inDropZone };
      case 'ADD_FILE_TO_LIST':
        return { ...state, fileList: state.fileList.concat(action.files) };
      default:
        return state;
    }
  };

  const [data, dispatch] = React.useReducer(
    reducer, { dropDepth: 0, inDropZone: false, fileList: [] }
  )

  return (
    <div className="App">
      <h1>React drag and drop component</h1>
      <DragAndDrop data={data} dispatch={dispatch} />

      <ol className="dropped-files">
        {data.fileList.map(f => {
          return (
            <li key={f.name}>{f.name}</li>
          )
        })}
      </ol>
    </div>
  );
}

export default App;
*/
