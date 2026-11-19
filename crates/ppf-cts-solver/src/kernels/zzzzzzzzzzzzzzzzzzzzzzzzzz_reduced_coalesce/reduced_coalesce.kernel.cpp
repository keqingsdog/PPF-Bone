#pragma once
// Generic exact-key coalescing. Only the preconditioning CSR changes storage.
// Equal-key run order is ascending original CSR index (two stable radix sorts).
[[seam::device_fn]] inline bool rsc_finite(float x){return x==x&&x<=3.402823466e38f&&x>=-3.402823466e38f;}
[[seam::device_fn]] inline void rsc_bad(compute::atomic_uint_t* status){compute::atomic_add(status+1u,1u);}

// mode0: exact original row IDs; mode1: row keys after stable column sort;
// mode2: complete cached CSR+mapping equality. No probabilistic hash/count key.
[[seam::entry(i)]]
[[seam::device_fn]] inline void rsc_prepare(
 const unsigned* offset,const unsigned* column,const unsigned* permutation,
 const unsigned* vertex_body,const unsigned* vertex_free,
 const unsigned* cached_offset,const unsigned* cached_column,
 const unsigned* cached_body,const unsigned* cached_free,
 unsigned* row_id,unsigned* row_key,compute::atomic_uint_t* status,
 unsigned rows,unsigned blocks,unsigned vertices,unsigned mode,unsigned i,unsigned extent){
 if(i>=extent)return;
 if(mode==0u){
  if(i>=rows)return;const unsigned begin=offset[i],end=offset[i+1u];
  if(begin>end||end>blocks||(i==0u&&begin!=0u)||(i+1u==rows&&end!=blocks)){rsc_bad(status);return;}
  for(unsigned k=begin;k<end;++k){if(column[k]>=rows)rsc_bad(status);row_id[k]=i;}
 }else if(mode==1u){
  if(i>=blocks)return;const unsigned p=permutation[i];
  if(p>=blocks){rsc_bad(status);return;}row_key[i]=row_id[p];
 }else if(mode==2u){
  bool changed=false;
  if(i<=rows)changed=changed||offset[i]!=cached_offset[i];
  if(i<blocks)changed=changed||column[i]!=cached_column[i];
  if(i<vertices)changed=changed||vertex_body[i]!=cached_body[i]||vertex_free[i]!=cached_free[i];
  if(changed)compute::atomic_add(status,1u);
 }else rsc_bad(status);
}

// Compose stable permutations. Flags get scanned using original ScanScratch;
// flags[blocks] is the sentinel total, not an extra logical block.
[[seam::entry(i)]]
[[seam::device_fn]] inline void rsc_segments(
 const unsigned* column,const unsigned* row_id,const unsigned* column_permutation,
 const unsigned* row_permutation,unsigned* permutation,unsigned* flags,
 compute::atomic_uint_t* status,unsigned rows,unsigned i,unsigned extent){
 if(i>=extent)return;
 const unsigned q=row_permutation[i];if(q>=extent){rsc_bad(status);return;}
 const unsigned p=column_permutation[q];if(p>=extent){rsc_bad(status);return;}
 const unsigned row=row_id[p],col=column[p];if(row>=rows||col>=rows){rsc_bad(status);return;}
 permutation[i]=p;unsigned first=1u;
 if(i!=0u){const unsigned pq=row_permutation[i-1u];if(pq>=extent){rsc_bad(status);return;}
  const unsigned pp=column_permutation[pq];if(pp>=extent){rsc_bad(status);return;}
  const unsigned pr=row_id[pp],pc=column[pp];
  if(pr>row||(pr==row&&pc>col)||(pr==row&&pc==col&&pp>=p)){rsc_bad(status);return;}
  first=(pr!=row||pc!=col)?1u:0u;
 }
 flags[i]=first;
}

// Fixed arithmetic shared by CPU simulation and CUDA warp entry. Lane l
// accumulates original positions start+l+32*t in increasing t; the32 partials
// then follow offsets16,8,4,2,1. No coefficient is skipped, including zeros.
[[seam::device_fn]] inline bool rsc_run_partial(const float* raw_value,
 const unsigned* permutation,unsigned start,unsigned n,unsigned blocks,
 unsigned component,unsigned lane,float& sum){
 sum=0.f;
 for(unsigned t=lane;t<n;){const unsigned pos=start+t,p=permutation[pos];
  if(p>=blocks||(pos>start&&p<=permutation[pos-1u]))return false;
  const float a=raw_value[9u*p+component];if(!rsc_finite(a))return false;
  sum+=a;if(!rsc_finite(sum))return false;
  // Avoid unsigned wrap even for a largest valid final stripe.
  if(n-t<=32u)break;t+=32u;
 }
 return true;
}

// mode0 builds unique topology and multiplicities from scanned head flags.
// mode1 refreshes all coefficients EVERY frozen Newton in original run order.
// mode2 optional actual GPU old/new CSR products on private diagnostic vectors.
// mode3 diagonal-block maxima, mode4 relative edge maxima; both write private
// graph weights only and leave the numerical operator unchanged.
[[seam::entry(i)]]
[[seam::device_fn]] inline void rsc_reduce(
 const unsigned* raw_offset,const unsigned* raw_column,const float* raw_value,
 const unsigned* row_id,const unsigned* permutation,const unsigned* scanned,
 unsigned* run_start,unsigned* run_length,compute::atomic_uint_t* unique_count,
 const unsigned* unique_offset,unsigned* unique_column,float* unique_value,
 const float* input,float* old_product,float* new_product,compute::atomic_uint_t* status,
 unsigned rows,unsigned blocks,unsigned unique_blocks,unsigned mode,unsigned i,unsigned extent){
 if(i>=extent)return;
 if(mode==0u){
  if(i>=blocks||scanned[i+1u]==scanned[i])return;
  const unsigned g=scanned[i],p=permutation[i];
  if(g>=unique_blocks||p>=blocks){rsc_bad(status);return;}
  const unsigned row=row_id[p],col=raw_column[p];if(row>=rows||col>=rows){rsc_bad(status);return;}
  unsigned end=i+1u;
  while(end<blocks){const unsigned q=permutation[end];if(q>=blocks){rsc_bad(status);return;}
   if(row_id[q]!=row||raw_column[q]!=col)break;++end;
  }
  run_start[g]=i;run_length[g]=end-i;unique_column[g]=col;compute::atomic_add(unique_count+row,1u);
 }else if(mode==1u){
  if(i>=unique_blocks)return;const unsigned start=run_start[i],n=run_length[i];
  if(n==0u||start>blocks||n>blocks-start){rsc_bad(status);return;}
  float out[9];
  for(unsigned k=0;k<9u;++k){float partial[32];
   for(unsigned lane=0;lane<32u;++lane)
    if(!rsc_run_partial(raw_value,permutation,start,n,blocks,k,lane,partial[lane])){rsc_bad(status);return;}
   for(unsigned delta=16u;delta!=0u;delta/=2u)
    for(unsigned lane=0;lane<delta;++lane){partial[lane]+=partial[lane+delta];
     if(!rsc_finite(partial[lane])){rsc_bad(status);return;}}
   out[k]=partial[0];
  }
  for(unsigned k=0;k<9u;++k)unique_value[9u*i+k]=out[k];
 }else if(mode==2u){
  if(i>=rows)return;
  for(unsigned r=0;r<3u;++r){float a=0.f,b=0.f;
   for(unsigned k=raw_offset[i];k<raw_offset[i+1u];++k){if(k>=blocks||raw_column[k]>=rows){rsc_bad(status);return;}
    for(unsigned c=0;c<3u;++c)a+=raw_value[9u*k+3u*c+r]*input[3u*raw_column[k]+c];}
   for(unsigned k=unique_offset[i];k<unique_offset[i+1u];++k){if(k>=unique_blocks||unique_column[k]>=rows){rsc_bad(status);return;}
    for(unsigned c=0;c<3u;++c)b+=unique_value[9u*k+3u*c+r]*input[3u*unique_column[k]+c];}
   if(!rsc_finite(a)||!rsc_finite(b)){rsc_bad(status);return;}old_product[3u*i+r]=a;new_product[3u*i+r]=b;
  }
 }else if(mode==3u){
  if(i>=rows)return;
  float diagonal=0.f;
  for(unsigned k=unique_offset[i];k<unique_offset[i+1u];++k)if(unique_column[k]==i)
   for(unsigned c=0u;c<9u;++c){float v=unique_value[9u*k+c];v=v<0.f?-v:v;if(v>diagonal)diagonal=v;}
  if(!(diagonal>0.f)||!rsc_finite(diagonal)){rsc_bad(status);return;}
  old_product[i]=diagonal;
 }else if(mode==4u){
  if(i>=rows)return;
  float left=fmath::sqrt(old_product[i]);
  if(!(left>0.f)||!rsc_finite(left)){rsc_bad(status);return;}
  for(unsigned k=unique_offset[i];k<unique_offset[i+1u];++k){
   unsigned col=unique_column[k];if(col>=rows){rsc_bad(status);return;}
   float right=fmath::sqrt(old_product[col]);
   if(!(right>0.f)||!rsc_finite(right)){rsc_bad(status);return;}
   float magnitude=0.f;
   for(unsigned c=0u;c<9u;++c){float v=unique_value[9u*k+c];v=v<0.f?-v:v;if(v>magnitude)magnitude=v;}
   // Divide by the larger scale first to avoid intermediate overflow.
   float large=left>right?left:right,small=left>right?right:left;
   float ratio=fmath::div(fmath::div(magnitude,large),small);
   if(!rsc_finite(ratio)){rsc_bad(status);return;}
   new_product[k]=ratio;
  }
 }else rsc_bad(status);
}
