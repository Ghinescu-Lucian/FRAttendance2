const video=document.getElementById("video") as HTMLVideoElement;
const canvas=document.getElementById("canvas") as HTMLCanvasElement;
const startCameraBtn=document.getElementById("startCameraBtn") as HTMLButtonElement;
const captureBtn=document.getElementById("captureBtn") as HTMLButtonElement;
const stopCameraBtn=document.getElementById("stopCameraBtn") as HTMLButtonElement;
const studentIdInput=document.getElementById("studentId") as HTMLInputElement;
const personNameInput=document.getElementById("personName") as HTMLInputElement;
const photoCountInput=document.getElementById("photoCount") as HTMLInputElement;
const statusEl=document.getElementById("status") as HTMLParagraphElement;
const countdownEl=document.getElementById("countdown") as HTMLDivElement;
const previewsEl=document.getElementById("previews") as HTMLDivElement;

let stream:MediaStream|null=null;

function setStatus(message:string):void{statusEl.textContent=message;}
function sleep(ms:number):Promise<void>{return new Promise(resolve=>window.setTimeout(resolve,ms));}

async function startCamera():Promise<void>{
  try{
    if(!navigator.mediaDevices?.getUserMedia){
      throw new Error("Camera API is not available. Use HTTPS or localhost.");
    }
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:1280},height:{ideal:720}},audio:false});
    video.srcObject=stream;
    startCameraBtn.disabled=true;
    captureBtn.disabled=false;
    stopCameraBtn.disabled=false;
    setStatus("Camera started. Put your face inside the green contour.");
  }catch(err){
    const error=err as Error;
    console.error(error);
    setStatus(`Camera error: ${error.name}: ${error.message}`);
  }
}

function stopCamera():void{
  if(stream){
    for(const track of stream.getTracks()) track.stop();
  }
  stream=null;
  video.srcObject=null;
  startCameraBtn.disabled=false;
  captureBtn.disabled=true;
  stopCameraBtn.disabled=true;
  setStatus("Camera stopped.");
}

function validateInputs():{studentId:string;personName:string}{
  const studentId=studentIdInput.value.trim();
  const personName=personNameInput.value.trim();
  if(!studentId) throw new Error("Student ID is required.");
  if(!personName) throw new Error("Person name is required.");
  return {studentId,personName};
}

async function captureImageBlob():Promise<Blob>{
  if(!video.videoWidth||!video.videoHeight) throw new Error("Video is not ready yet.");
  canvas.width=video.videoWidth;
  canvas.height=video.videoHeight;
  const ctx=canvas.getContext("2d");
  if(!ctx) throw new Error("Could not create canvas context.");
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  return new Promise((resolve,reject)=>{
    canvas.toBlob(blob=>{
      if(!blob){reject(new Error("Could not create image blob."));return;}
      resolve(blob);
    },"image/jpeg",0.92);
  });
}

async function uploadImage(blob:Blob,frameIndex:number):Promise<void>{
  const {studentId,personName}=validateInputs();
  const formData=new FormData();
  formData.append("studentId",studentId);
  formData.append("name",personName);
  formData.append("frameIndex",String(frameIndex));
  formData.append("image",blob,`frame_${frameIndex}.jpg`);
  const response=await fetch("/api/upload-face",{method:"POST",body:formData});
  const data=await response.json().catch(()=>null);
  if(!response.ok||!data?.ok) throw new Error(data?.error||`Upload failed with status ${response.status}`);
}

async function completeEnrollment():Promise<any>{
  const {studentId,personName}=validateInputs();
  const response=await fetch("/api/complete-enrollment",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({studentId,name:personName})});
  if(!response.ok) throw new Error(`Complete enrollment failed with status ${response.status}`);
  return await response.json();
}

function addPreview(blob:Blob):void{
  const img=document.createElement("img");
  img.src=URL.createObjectURL(blob);
  previewsEl.prepend(img);
}

async function captureSequence():Promise<void>{
  captureBtn.disabled=true;
  previewsEl.innerHTML="";
  try{
    validateInputs();
    const count=Math.max(1,Math.min(10,Number(photoCountInput.value||5)));
    for(let i=1;i<=count;i++){
      countdownEl.textContent=String(i);
      setStatus(`Capturing photo ${i}/${count}. Keep your face inside the green contour.`);
      await sleep(650);
      const blob=await captureImageBlob();
      addPreview(blob);
      await uploadImage(blob,i);
      await sleep(250);
    }
    countdownEl.textContent="";
    const summary=await completeEnrollment();
    setStatus(`Done. Server sees ${summary.savedImages} image(s) for ${summary.name}. Saved to: ${summary.uploadDir}`);
  }catch(err){
    const error=err as Error;
    console.error(error);
    countdownEl.textContent="";
    setStatus(`Error: ${error.message}`);
  }finally{
    captureBtn.disabled=false;
  }
}

startCameraBtn.addEventListener("click",()=>void startCamera());
stopCameraBtn.addEventListener("click",stopCamera);
captureBtn.addEventListener("click",()=>void captureSequence());
