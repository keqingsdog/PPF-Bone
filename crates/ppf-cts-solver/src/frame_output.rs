//! Optional owned-CPU frame publisher. No simulation/GPU references cross threads.
use std::{io::Write, path::{Path,PathBuf}, sync::{Arc,Mutex,Condvar,mpsc::{self,SyncSender}}, thread::{self,JoinHandle}};
const BYTE_LIMIT:usize=64*1024*1024;
pub(crate) struct FramePacket {
    pub frame:i32, pub vertices:Vec<u8>, pub statistics:Option<Vec<u8>>,
    pub rigid_pose:Option<Vec<u8>>, pub fem_pose:Option<Vec<u8>>, pub display_pins:Option<Vec<u8>>,
    pub keep_verts:i32, pub progress:Option<f64>,
}
impl FramePacket {
    fn bytes(&self)->usize { self.vertices.capacity()+[&self.statistics,&self.rigid_pose,&self.fem_pose,&self.display_pins].iter().map(|v|v.as_ref().map_or(0,|b|b.capacity())).sum::<usize>() }
}
#[derive(Default)]
struct State { pending:usize, peak:usize, published:u64, fallback:u64, final_frame:Option<i32>, error:Option<String> }
type Shared=Arc<(Mutex<State>,Condvar)>;
enum Message { Frame(FramePacket,usize), Barrier(mpsc::Sender<()>) }
pub(crate) struct Publisher { output:PathBuf, tx:Option<SyncSender<Message>>, worker:Option<JoinHandle<()>>, shared:Shared, last:Option<i32>, finished:bool }
fn failed(shared:&Shared,error:String) { let (lock,cv)=&**shared; let mut s=lock.lock().unwrap_or_else(|p|p.into_inner()); if s.error.is_none(){s.error=Some(error);} cv.notify_all(); }
fn publish(output:&Path,p:&FramePacket)->Result<(),String> {
    let name=ppf_cts_formats::files::vert_filename(p.frame);
    let tmp=output.join(format!("{name}.tmp"));
    let mut writer=std::io::BufWriter::with_capacity(crate::IO_BUF_CAPACITY,std::fs::OpenOptions::new().write(true).create(true).truncate(true).open(&tmp).map_err(|e|e.to_string())?);
    writer.write_all(&p.vertices).map_err(|e|e.to_string())?;
    writer.flush().map_err(|e|e.to_string())?;
    if let Some(b)=&p.statistics { crate::statistics::publish_encoded(output,p.frame as u32,b); }
    // Keep the original vertex flush/rename semantics (no added or removed fsync).
    std::fs::rename(&tmp,output.join(name)).map_err(|e|e.to_string())?;
    #[cfg(neutral_driver)]
    {
        if let Some(b)=&p.rigid_pose {
            let mut cursor=std::io::Cursor::new(b.as_slice());
            let f:crate::pose_output::PoseFrame=ciborium::from_reader(&mut cursor).map_err(|e|e.to_string())?;
            if cursor.position()!=b.len() as u64 || f.frame_id!=p.frame as u32 {return Err("rigid packet identity mismatch".into());}
            if crate::pose_output::encode(&f,&p.vertices)?!=*b {return Err("rigid packet encoding changed".into());}
            crate::pose_output::publish(output,&f,&p.vertices)?;
        }
        if let Some(b)=&p.fem_pose {
            let mut cursor=std::io::Cursor::new(b.as_slice());
            let f:crate::soft_pose_output::PoseFrame=ciborium::from_reader(&mut cursor).map_err(|e|e.to_string())?;
            if cursor.position()!=b.len() as u64 || f.frame_id!=p.frame as u32 {return Err("FEM packet identity mismatch".into());}
            if crate::soft_pose_output::encode(&f,&p.vertices)?!=*b {return Err("FEM packet encoding changed".into());}
            crate::soft_pose_output::publish(output,&f,&p.vertices)?;
        }
    }
    let dir=output.to_str().ok_or("output path is not UTF-8")?;
    crate::remove_old_files(dir,ppf_cts_formats::files::VERT_PREFIX,ppf_cts_formats::files::VERT_SUFFIX,p.keep_verts,p.frame);
    if let Some(b)=&p.display_pins {
        let name=ppf_cts_formats::files::display_pin_filename(p.frame);
        let tmp=output.join(format!("{name}.tmp"));
        std::fs::write(&tmp,b).map_err(|e|e.to_string())?;
        std::fs::rename(tmp,output.join(name)).map_err(|e|e.to_string())?;
        crate::remove_old_files(dir,ppf_cts_formats::files::DISPLAY_PIN_PREFIX,ppf_cts_formats::files::DISPLAY_PIN_SUFFIX,p.keep_verts,p.frame);
    }
    if let Some(t)=p.progress { crate::status_writer::progress(ppf_cts_formats::status::Phase::Running,p.frame,t); }
    Ok(())
}
// Both the FIFO worker and the oversized synchronous fallback use this path.
fn commit(output:&Path,p:&FramePacket,shared:&Shared)->Result<(),String> {
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||publish(output,p)))
        .unwrap_or_else(|_|Err("frame publisher panicked".into()));
    match result {
        Ok(())=>{let mut s=shared.0.lock().unwrap_or_else(|p|p.into_inner());s.published+=1;s.final_frame=Some(p.frame);Ok(())},
        Err(e)=>{failed(shared,e.clone());Err(e)}
    }
}
impl Publisher {
    pub fn new(output:&str)->Self {
        let output=PathBuf::from(output); let directory=output.clone();
        let shared=Arc::new((Mutex::new(State::default()),Condvar::new()));let state=shared.clone();
        let (tx,rx)=mpsc::sync_channel(2);
        let worker=thread::Builder::new().name("ppf-frame-publisher".into()).spawn(move||{
            let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                while let Ok(message)=rx.recv() { match message {
                    Message::Frame(packet,n)=>{
                        let result=commit(&directory,&packet,&state);
                        drop(packet); // release bytes before releasing budget
                        let (lock,cv)=&*state;let mut s=lock.lock().unwrap_or_else(|p|p.into_inner());s.pending-=n;cv.notify_all();drop(s);
                        if result.is_err(){break;}
                    },
                    Message::Barrier(ack)=>{let _=ack.send(());}
                }}
            }));
            if result.is_err(){failed(&state,"frame publisher thread panicked".into());}
        }).expect("create frame publisher");
        Self{output,tx:Some(tx),worker:Some(worker),shared,last:None,finished:false}
    }
    pub fn check(&self)->Result<(),String> {
        if let Some(e)=&self.shared.0.lock().unwrap_or_else(|p|p.into_inner()).error{return Err(e.clone());}
        if !self.finished && self.worker.as_ref().is_some_and(|w|w.is_finished()){return Err("frame publisher exited unexpectedly".into());}
        Ok(())
    }
    pub fn barrier(&mut self)->Result<(),String> {
        self.check()?; if self.finished{return Ok(());}
        let (tx,rx)=mpsc::channel();
        self.tx.as_ref().ok_or("publisher is closed")?.send(Message::Barrier(tx)).map_err(|_|"frame publisher disconnected".to_string())?;
        rx.recv().map_err(|_|"frame publisher barrier disconnected".to_string())?;self.check()
    }
    pub fn submit(&mut self,p:FramePacket)->Result<(),String> {
        self.check()?;if self.finished{return Err("frame submitted after finish".into());}
        if self.last.is_some_and(|v|p.frame!=v+1){return Err("nonconsecutive output frame".into());}
        let n=p.bytes(); let frame=p.frame;
        if n>BYTE_LIMIT {
            self.barrier()?;
            commit(&self.output,&p,&self.shared)?;
            self.shared.0.lock().unwrap_or_else(|p|p.into_inner()).fallback+=1;
        } else {
            let (lock,cv)=&*self.shared;let mut s=lock.lock().unwrap_or_else(|p|p.into_inner());
            while s.pending+n>BYTE_LIMIT && s.error.is_none(){s=cv.wait(s).unwrap_or_else(|p|p.into_inner());}
            if let Some(e)=&s.error{return Err(e.clone());}
            s.pending+=n;s.peak=s.peak.max(s.pending);drop(s);
            if self.tx.as_ref().ok_or("publisher is closed")?.send(Message::Frame(p,n)).is_err(){self.check()?;return Err("frame publisher disconnected".into());}
            self.last=Some(frame);
            return Ok(());
        }
        self.last=Some(p.frame);Ok(())
    }
    pub fn finish(&mut self)->Result<(),String> {
        if self.finished{return self.check();}
        self.tx.take();
        if let Some(w)=self.worker.take(){if w.join().is_err(){failed(&self.shared,"frame publisher join panic".into());}}
        self.finished=true;self.check()?;
        let s=self.shared.0.lock().unwrap_or_else(|p|p.into_inner());
        if s.pending!=0{return Err("frame publisher drained with pending bytes".into());}
        let receipt=serde_json::json!({"kind":"ppf.async_output.receipt","version":1,"mode":"async-output","queue_capacity":2,"byte_budget":BYTE_LIMIT,"peak_queued_inflight_bytes":s.peak,"published_frames":s.published,"oversized_sync_fallbacks":s.fallback,"final_frame":s.final_frame,"queue_drained":true});
        std::fs::write(self.output.join("async_output_receipt.json"),serde_json::to_vec_pretty(&receipt).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
    }
}
impl Drop for Publisher { fn drop(&mut self){self.tx.take();if let Some(w)=self.worker.take(){let _=w.join();}} }
