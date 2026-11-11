"""Counts for the report from the same visible forms as the floor receipt."""

def view_counts(snapshot, level):
    """Whole source forms and the floor's final visible replacements, without counting swallowed rungs."""
    from ouroboros import memory_form_facts, memory_view_conversations as conv
    from ouroboros.memory_view_work import (folded_work_rows, task_line, works, conversation_task_lines,
                                          task_gist_line, task_gists_before_work_lines)
    import re
    from collections import Counter

    gone = {step: set(ids) for step, ids in level.addressed}
    room, legacy = snapshot.room or {}, "legacy, a helper's"
    lined = {g["key"] for g in snapshot.conversations if conv.group_form(g, gone) or g["key"] in gone.get("F6a", ())}
    rooms = [(group, item) for group in snapshot.conversations for item in group['rooms']]
    words = [(group, word) for group, item in rooms for word in item['words']]
    hidden = lambda group, word: (word['id'] in gone.get('F7', ()) if word['owner_side'] and not word.get('again')
                                  else group['key'] in lined or word['id'] in gone.get('F6w', ()) or 'bare:' + word['id'] in gone.get('F6w', ()))
    summary = lambda line: '— summary: «' in line or '; «' in line
    early = [item for item in room.get('lane2', ()) if item['first'] in gone.get('F1', ())]
    folded = {key for key, _rows in works(room) if any('work:' + key + suffix in gone.get('F2', ())
                                                     for suffix in (':progress', ':final'))}
    lines = [task_gist_line(t) for t in task_gists_before_work_lines(room, gone)] + [
             task_line(item, folded=not gone.get('F7a') and item.get('task') in folded)
             for item in room.get('lane2', ()) if item not in early] + [
        line for group in snapshot.conversations if group['key'] not in lined
        for task_lines in conversation_task_lines(group).values() for line in task_lines]
    told = Counter(legacy if e['kind'] == 'legacy' else e['kind'] + (' draft' if e.get('status') == 'draft' else '')
                   for e in snapshot.story if not e.get('told_by') and (e['room_id'] not in gone.get('F3', ()) if
                   e['kind'] == 'legacy' else e['id'] not in gone.get('F5', set()) | gone.get('F5l', set())))
    told.update('standing part member' if item.get('standing') else item.get('kind', legacy)
                for item in [*room.get('legacy', ()), *room.get('under_parts', ())]
                if item.get('standing') or item['id'] not in gone.get('F4', ()))
    told.update({'task summary': sum(map(summary, lines))})
    silent = [(group['key'] in lined, re.search(r'(\d+) words nobody considered', i['silent']))
              for group, i in rooms if not i['engaged']]
    host = Counter({'task line': len(lines) - told['task summary'], 'earlier tasks': int(bool(early)),
                    'repeat': sum(bool(w.get('again')) and not hidden(g, w) for g, w in words),
                    'silent room': sum(not in_line for in_line, _found in silent)})
    unconsidered = Counter('as a line' if hidden(g, w) else 'as text' for g, w in words if w['rank'] == 2)
    unconsidered['as a line'] += sum(int(found.group(1)) for _in_line, found in silent if found)
    forms = memory_form_facts.form_counts(snapshot, level)
    told.update({'fold_read' if name == 'read and folded' else name: fact['lines']
                 for name, fact in forms['retold'].items() if name != 'task gists'})
    emergency = Counter({name: fact['lines'] for name, fact in forms['emergency'].items()})
    addressed = {ident.removeprefix('bare:') for step in ('F2', 'F7', 'F7p') for ident in gone.get(step, ())}
    addressed |= folded_work_rows(room, gone)[0]
    rows = [*room.get('lane1', ()), *(item for item in room.get('project_words') or () if item.get('address'))]
    return {'retellings': dict(+told), 'host_facts': dict(+host), 'unconsidered': dict(+unconsidered),
            'emergency': dict(+emergency), 'reduced_forms': forms, 'rows_as_words': sum(item['address'] not in addressed for item in rows)
            + len(room.get('origins', ())) + sum(not hidden(g, w) and not w.get('again') for g, w in words)}
