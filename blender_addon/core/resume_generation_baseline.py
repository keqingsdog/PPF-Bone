"""Validate read-only server proof for a missing resume-generation status."""
FIELDS=frozenset(('kind','version','name','root','upload_id','param_hash','data_hash','launch_id'))

def prior_generation(response, expected, project):
    if not isinstance(response,dict) or response.get('resume_generation_error'):
        return None
    proof=response.get('resume_generation_baseline')
    if not isinstance(proof,dict) or set(proof)!=FIELDS:
        return None
    if proof.get('kind')!='ppf.resume_generation_baseline' or type(proof.get('version')) is not int or proof['version']!=1:
        return None
    if not isinstance(project,str) or not project or proof.get('name')!=project:
        return None
    for key in ('root','upload_id','param_hash','data_hash'):
        if not isinstance(expected.get(key),str) or not expected[key] or response.get(key)!=expected[key] or proof.get(key)!=expected[key]:
            return None
    launch=proof.get('launch_id')
    if not isinstance(launch,str) or not launch or len(launch)>128:
        return None
    return {'launch_id':launch}
