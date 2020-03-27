let exec = require('child_process').exec;
let express = require('express');
let app = express();
let multer = require('multer')
let cors = require('cors');

let current_img = ""
const FILEPATH = 'public/img'
/* storage definition */
let storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, FILEPATH)
    },
    filename: function (req, file, cb) {
        let filename = Date.now() + '_' +file.originalname;
        current_img = filename
        cb(null, filename)
    }
})

let upload = multer({ storage: storage }).array('file')

/* POST request */
app.use(cors())
app.use(FILEPATH, express.static("img"))

app.get('/',function(req,res){
    return res.send('Hello Server')
})

app.get('/getimage',function(req,res){
    return res.send(current_image)
})

app.post('/upload',function(req, res) {
    upload(req, res, function (err) {
        // A Multer error occurred when uploading or An unknown error occurred when uploading.
        if (err instanceof multer.MulterError || err) {
            return res.status(500).json(err)
        }
        let img_result = 'res.png'
        let cmd = `convert -resize 48x48 '${FILEPATH}/${current_img}' '${FILEPATH}/${img_result}'`
        console.log(cmd)
        exec(cmd, (error, stdout, stderr) => {
            if (error) {
                console.log(`error: ${error.message}`);
                return;
            }
            if (stderr) {
                console.log(`stderr: ${stderr}`);
                return;
            }
            console.log(`stdout: ${stdout}`);
            return res.status(200).send(img_result) // Everything went fine.
        });

      })
});

app.listen(8000, function() {
    console.log('App running on port 8000');
});
