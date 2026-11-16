"""Compare embedded Ascend device executable sections in immutable binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from datetime import datetime,timezone


def sha(data):return hashlib.sha256(data).hexdigest()


def device_elfs(binary,extract=None,label='device'):
    data=binary.read_bytes();pos=0;result=[]
    while True:
        start=data.find(b'\x7fELF',pos)
        if start<0:break
        pos=start+4
        if start+64>len(data) or data[start+4:start+6]!=bytes([2,1]):continue
        machine=struct.unpack_from('<H',data,start+18)[0]
        if machine!=4137:continue
        shoff=struct.unpack_from('<Q',data,start+40)[0]
        entsize,count,names_index=struct.unpack_from('<HHH',data,start+58)
        if entsize!=64 or not count or names_index>=count:continue
        end=shoff+count*entsize
        if start+end>len(data):continue
        sections=[struct.unpack_from('<IIQQQQIIQQ',data,start+shoff+i*entsize) for i in range(count)]
        names=sections[names_index]
        strings=data[start+names[4]:start+names[4]+names[5]]
        executable=[]
        for section in sections:
            nameoff,kind,flags,_,offset,size,*_=section
            if kind==8:continue  # SHT_NOBITS is not part of the file.
            end=max(end,offset+size)
            if not flags&4:continue
            name=strings[nameoff:].split(b'\0',1)[0].decode(errors='replace')
            payload=data[start+offset:start+offset+size]
            if len(payload)!=size:raise ValueError('Truncated executable section')
            executable.append(dict(name=name,size=size,sha256=sha(payload)))
        if not executable:raise ValueError('Device ELF has no executable sections')
        raw=data[start:start+end]
        row=dict(offset=start,machine=machine,size=end,sha256=sha(raw),executable_sections=executable)
        if extract:
            path=extract/f'{label}-{len(result)}.elf';path.write_bytes(raw);row['extracted']=str(path)
        result.append(row)
    if not result:raise ValueError('No supported embedded device ELF found')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary',action='append',required=True,help='label=absolute-binary')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--extract-dir',type=Path)
    args=ap.parse_args();records={}
    if args.extract_dir:args.extract_dir.mkdir(parents=True,exist_ok=True)
    for spec in args.binary:
        label,path=spec.split('=',1);binary=Path(path).resolve(strict=True)
        if not label.replace('_','').isalnum():raise ValueError('Use simple labels')
        source=binary.parent.parent/'kernel.asc'
        records[label]=dict(binary=str(binary),binary_sha256=sha(binary.read_bytes()),
                            source=str(source),source_sha256=sha(source.read_bytes()),
                            elfs=device_elfs(binary,args.extract_dir,label))
    base=next(iter(records));comparisons={}
    for label,record in records.items():
        if label==base:continue
        left=records[base]['elfs'];right=record['elfs'];different=[]
        for index,(a,b) in enumerate(zip(left,right)):
            aa={s['name']:(s['size'],s['sha256']) for s in a['executable_sections']}
            bb={s['name']:(s['size'],s['sha256']) for s in b['executable_sections']}
            for name in sorted(aa.keys()|bb.keys()):
                if aa.get(name)!=bb.get(name):different.append(dict(elf=index,section=name,left=aa.get(name),right=bb.get(name)))
        comparisons[label]=dict(baseline=base,same_device_elf_count=len(left)==len(right),
                                all_device_elf_bytes_identical=len(left)==len(right) and all(a['sha256']==b['sha256'] for a,b in zip(left,right)),
                                all_executable_sections_identical=len(left)==len(right) and not different,
                                different_executable_sections=different)
    report=dict(created_utc=datetime.now(timezone.utc).isoformat(),records=records,comparisons=comparisons,
                limits=['Compares device executable sections, not host launch arguments or runtime schedules.',
                        'Identical instructions do not imply identical measured latency or unchanged TBest.'])
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({label:{k:v for k,v in d.items() if k!='different_executable_sections'}|dict(different_section_count=len(d['different_executable_sections'])) for label,d in comparisons.items()},indent=2))


if __name__=='__main__':main()
