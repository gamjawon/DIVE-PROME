// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'place_search_model.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$PlaceSearchResponse {

 List<Location> get documents;
/// Create a copy of PlaceSearchResponse
/// with the given fields replaced by the non-null parameter values.
@JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
$PlaceSearchResponseCopyWith<PlaceSearchResponse> get copyWith => _$PlaceSearchResponseCopyWithImpl<PlaceSearchResponse>(this as PlaceSearchResponse, _$identity);

  /// Serializes this PlaceSearchResponse to a JSON map.
  Map<String, dynamic> toJson();


@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is PlaceSearchResponse&&const DeepCollectionEquality().equals(other.documents, documents));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(documents));

@override
String toString() {
  return 'PlaceSearchResponse(documents: $documents)';
}


}

/// @nodoc
abstract mixin class $PlaceSearchResponseCopyWith<$Res>  {
  factory $PlaceSearchResponseCopyWith(PlaceSearchResponse value, $Res Function(PlaceSearchResponse) _then) = _$PlaceSearchResponseCopyWithImpl;
@useResult
$Res call({
 List<Location> documents
});




}
/// @nodoc
class _$PlaceSearchResponseCopyWithImpl<$Res>
    implements $PlaceSearchResponseCopyWith<$Res> {
  _$PlaceSearchResponseCopyWithImpl(this._self, this._then);

  final PlaceSearchResponse _self;
  final $Res Function(PlaceSearchResponse) _then;

/// Create a copy of PlaceSearchResponse
/// with the given fields replaced by the non-null parameter values.
@pragma('vm:prefer-inline') @override $Res call({Object? documents = null,}) {
  return _then(_self.copyWith(
documents: null == documents ? _self.documents : documents // ignore: cast_nullable_to_non_nullable
as List<Location>,
  ));
}

}


/// Adds pattern-matching-related methods to [PlaceSearchResponse].
extension PlaceSearchResponsePatterns on PlaceSearchResponse {
/// A variant of `map` that fallback to returning `orElse`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeMap<TResult extends Object?>(TResult Function( _PlaceSearchResponse value)?  $default,{required TResult orElse(),}){
final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// Callbacks receives the raw object, upcasted.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case final Subclass2 value:
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult map<TResult extends Object?>(TResult Function( _PlaceSearchResponse value)  $default,){
final _that = this;
switch (_that) {
case _PlaceSearchResponse():
return $default(_that);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `map` that fallback to returning `null`.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case final Subclass value:
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? mapOrNull<TResult extends Object?>(TResult? Function( _PlaceSearchResponse value)?  $default,){
final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that);case _:
  return null;

}
}
/// A variant of `when` that fallback to an `orElse` callback.
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return orElse();
/// }
/// ```

@optionalTypeArgs TResult maybeWhen<TResult extends Object?>(TResult Function( List<Location> documents)?  $default,{required TResult orElse(),}) {final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that.documents);case _:
  return orElse();

}
}
/// A `switch`-like method, using callbacks.
///
/// As opposed to `map`, this offers destructuring.
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case Subclass2(:final field2):
///     return ...;
/// }
/// ```

@optionalTypeArgs TResult when<TResult extends Object?>(TResult Function( List<Location> documents)  $default,) {final _that = this;
switch (_that) {
case _PlaceSearchResponse():
return $default(_that.documents);case _:
  throw StateError('Unexpected subclass');

}
}
/// A variant of `when` that fallback to returning `null`
///
/// It is equivalent to doing:
/// ```dart
/// switch (sealedClass) {
///   case Subclass(:final field):
///     return ...;
///   case _:
///     return null;
/// }
/// ```

@optionalTypeArgs TResult? whenOrNull<TResult extends Object?>(TResult? Function( List<Location> documents)?  $default,) {final _that = this;
switch (_that) {
case _PlaceSearchResponse() when $default != null:
return $default(_that.documents);case _:
  return null;

}
}

}

/// @nodoc
@JsonSerializable()

class _PlaceSearchResponse implements PlaceSearchResponse {
  const _PlaceSearchResponse({required final  List<Location> documents}): _documents = documents;
  factory _PlaceSearchResponse.fromJson(Map<String, dynamic> json) => _$PlaceSearchResponseFromJson(json);

 final  List<Location> _documents;
@override List<Location> get documents {
  if (_documents is EqualUnmodifiableListView) return _documents;
  // ignore: implicit_dynamic_type
  return EqualUnmodifiableListView(_documents);
}


/// Create a copy of PlaceSearchResponse
/// with the given fields replaced by the non-null parameter values.
@override @JsonKey(includeFromJson: false, includeToJson: false)
@pragma('vm:prefer-inline')
_$PlaceSearchResponseCopyWith<_PlaceSearchResponse> get copyWith => __$PlaceSearchResponseCopyWithImpl<_PlaceSearchResponse>(this, _$identity);

@override
Map<String, dynamic> toJson() {
  return _$PlaceSearchResponseToJson(this, );
}

@override
bool operator ==(Object other) {
  return identical(this, other) || (other.runtimeType == runtimeType&&other is _PlaceSearchResponse&&const DeepCollectionEquality().equals(other._documents, _documents));
}

@JsonKey(includeFromJson: false, includeToJson: false)
@override
int get hashCode => Object.hash(runtimeType,const DeepCollectionEquality().hash(_documents));

@override
String toString() {
  return 'PlaceSearchResponse(documents: $documents)';
}


}

/// @nodoc
abstract mixin class _$PlaceSearchResponseCopyWith<$Res> implements $PlaceSearchResponseCopyWith<$Res> {
  factory _$PlaceSearchResponseCopyWith(_PlaceSearchResponse value, $Res Function(_PlaceSearchResponse) _then) = __$PlaceSearchResponseCopyWithImpl;
@override @useResult
$Res call({
 List<Location> documents
});




}
/// @nodoc
class __$PlaceSearchResponseCopyWithImpl<$Res>
    implements _$PlaceSearchResponseCopyWith<$Res> {
  __$PlaceSearchResponseCopyWithImpl(this._self, this._then);

  final _PlaceSearchResponse _self;
  final $Res Function(_PlaceSearchResponse) _then;

/// Create a copy of PlaceSearchResponse
/// with the given fields replaced by the non-null parameter values.
@override @pragma('vm:prefer-inline') $Res call({Object? documents = null,}) {
  return _then(_PlaceSearchResponse(
documents: null == documents ? _self._documents : documents // ignore: cast_nullable_to_non_nullable
as List<Location>,
  ));
}


}

// dart format on
